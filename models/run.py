#!/usr/bin/env python3

import argparse
from contextlib import contextmanager
import datetime
from itertools import chain, groupby, repeat
import logging
import math
import os
import pickle
import signal
import subprocess
import sys
import tempfile

import numpy as np
import h5py
from scipy.interpolate import interp1d
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage.interpolation import rotate, shift

import freestream
import frzout
import JetCalc.LeadingParton as JLP
from JetCalc.ExpCut import cuts as JEC

def run_cmd(*args):
	"""
	Run and log a subprocess.

	"""
	cmd = ' '.join(args)
	logging.info('running command: %s', cmd)

	try:
		proc = subprocess.run(
			cmd.split(), check=True,
			stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
			universal_newlines=True
		)
		print(proc.stdout)
	except subprocess.CalledProcessError as e:
		logging.error(
			'command failed with status %d:\n%s',
			e.returncode, e.output.strip('\n')
		)
		raise
	else:
		logging.debug(
			'command completed successfully:\n%s',
			proc.stdout.strip('\n')
		)
		return proc

def read_text_file(filename):
	"""
	Read a text file into a nested list of bytes objects,
	skipping comment lines (#).

	"""
	with open(filename, 'rb') as f:
		return [l.split() for l in f if not l.startswith(b'#')]

class Parser(argparse.ArgumentParser):
	"""
	ArgumentParser that parses files with 'key = value' lines.

	"""
	def __init__(self, *args, fromfile_prefix_chars='@', **kwargs):
		super().__init__(
			*args, fromfile_prefix_chars=fromfile_prefix_chars, **kwargs
		)

	def convert_arg_line_to_args(self, arg_line):
		# split each line on = and prepend prefix chars to first arg so it is
		# parsed as a long option
		args = [i.strip() for i in arg_line.split('=', maxsplit=1)]
		args[0] = 2*self.prefix_chars[0] + args[0]
		return args


parser = Parser(
	usage=''.join('\n  %(prog)s ' + i for i in [
		'[options] <results_file>',
		'checkpoint <checkpoint_file>',
		'-h | --help',
	]),
	description='''
Run relativistic heavy-ion collision events.

In the first form, run events according to the given options (below) and write
results to binary file <results_file>.

In the second form, run the event saved in <checkpoint_file>, previously
created by using the --checkpoint option and interrupting an event in progress.
''',
	formatter_class=argparse.RawDescriptionHelpFormatter
)


def parse_args_checkpoint():
	"""
	Parse command line arguments according to the parser usage info.  Return a
	tuple (args, ic) where `args` is a normal argparse.Namespace and `ic` is
	either None or an np.array of the checkpointed initial condition.

	First, check for the special "checkpoint" form, and if found, load and
	return the args and checkpoint initial condition from the specified file.
	If not, let the parser object handle everything.

	This is a little hacky but it works fine.  Truth is, argparse can't do
	exactly what I want here.  I suppose `docopt` might be a better option, but
	it's not worth the effort to rewrite everything.

	"""
	def usage():
		parser.print_usage(sys.stderr)
		sys.exit(2)

	if len(sys.argv) == 1:
		usage()

	if sys.argv[1] == 'checkpoint':
		if len(sys.argv) != 3:
			usage()

		path = sys.argv[2]

		try:
			with open(path, 'rb') as f:
				args, ic, nb = pickle.load(f)
		except Exception as e:
			msg = '{}: {}'.format(type(e).__name__, e)
			if path not in msg:
				msg += ": '{}'".format(path)
			sys.exit(msg)

		# as a simple integrity check, require that the checkpoint file is
		# actually the file specified in the checkpointed args
		if os.path.abspath(path) != args.checkpoint:
			sys.exit(
				"checkpoint file path '{}' does not match saved path '{}'"
				.format(path, args.checkpoint)
			)

		return args, np.array([ic, nb])

	return parser.parse_args(), None


parser.add_argument(
	'results', type=os.path.abspath,
	help=argparse.SUPPRESS
)
parser.add_argument(
	'--buffering', type=int, default=0, metavar='INT',
	help='results file buffer size in bytes (default: no buffering)'
)
parser.add_argument(
	'--nevents', type=int, metavar='INT',
	help='number of events to run (default: run until interrupted)'
)
parser.add_argument(
	'--avg-ic', default='off', metavar='VAR',
	help='if on, generate 500 IC events in the centrality bin and take average'
)
parser.add_argument(
	'--rankvar', metavar='VAR',
	help='environment variable containing process rank'
)
parser.add_argument(
	'--rankfmt', metavar='FMT',
	help='format string for rank integer'
)
parser.add_argument(
	'--tmpdir', type=os.path.abspath, metavar='PATH',
	help='temporary directory (default: {})'.format(tempfile.gettempdir())
)
parser.add_argument(
	'--checkpoint', type=os.path.abspath, metavar='PATH',
	help='checkpoint file [pickle format]'
)
parser.add_argument(
	'--particles', type=os.path.abspath, metavar='PATH',
	help='raw particle data file (default: do not save)'
)
parser.add_argument(
	'--logfile', type=os.path.abspath, metavar='PATH',
	help='log file (default: stdout)'
)
parser.add_argument(
	'--loglevel', choices={'debug', 'info', 'warning', 'error', 'critical'},
	default='info',
	help='log level (default: %(default)s)'
)
parser.add_argument(
	'--trento-args', default='Pb Pb', metavar='ARGS',
	help="arguments passed to trento (default: '%(default)s')"
)
parser.add_argument(
	'--tau-fs', type=float, default=.5, metavar='FLOAT',
	help='free streaming time [fm] (default: %(default)s fm)'
)
parser.add_argument(
	'--xi-fs', type=float, default=.5, metavar='FLOAT',
	help='energy loss staring time / tau-fs'
)
parser.add_argument(
	'--hydro-args', default='', metavar='ARGS',
	help='arguments passed to osu-hydro (default: empty)'
)
parser.add_argument(
	'--Tswitch', type=float, default=.150, metavar='FLOAT',
	help='particlization temperature [GeV] (default: %(default).3f GeV)'
)
parser.add_argument(
	'--NPythiaEvents', type=int, default=100000, metavar='INT',
	help='number of pythia events'
)
parser.add_argument(
	'--sqrts', type=float, default=5020, metavar='FLOAT',
	help='center-of-mass energy'
)
parser.add_argument(
	'--A', type=float, default=1.0, metavar='FLOAT',
	help='diffusion parameter #1'
)
parser.add_argument(
	'--B', type=float, default=0., metavar='FLOAT',
	help='diffusion parameter #2'
)
parser.add_argument(
	'--mu', type=float, default=1.0, metavar='FLOAT',
	help='running coupling scale'
)
parser.add_argument(
	'--afix', type=float, default=-1, metavar='FLOAT',
	help='fixed coupling constant, -1 --> running'
)
parser.add_argument(
	'--proj', type=str, default="Pb", metavar='STR',
	help='projectile'
)
parser.add_argument(
	'--targ', type=str, default="Pb", metavar='STR',
	help='target'
)
# pid and datatype
species = {
		'light':
			[('pion', 211),
			('kaon', 321),
			('proton', 2212),
			('Lambda', 3122),
			('Sigma0', 3212),
			('Xi', 3312),
			('Omega', 3334)],
		'heavy':
			[('D+', 411), 
			('D0', 421),
			('D*+', 10411), 
			('D0*', 10421)],
			}
# fully specify numeric data types, including endianness and size, to
# ensure consistency across all machines
float_t = '<f8'
int_t = '<i8'
complex_t = '<c16'
# results "array" (one element)
# to be overwritten for each event
result_dtype=[
	########### Initial condition #####################
	('initial_entropy', float_t),
	('Npart', float_t),
	('Ncoll', float_t),
	########### Soft part #############################
	('nsamples', int_t),
	('dNch_deta', float_t),
	('dET_deta', float_t),
	('dN_dy', [(s, float_t) for (s, _) in species.get('light')]),
	('mean_pT', [(s, float_t) for (s, _) in species.get('light')]),
	('pT_fluct', [('N', int_t), ('sum_pT', float_t), ('sum_pTsq', float_t)]),
	('Qn_soft', [('M', int_t), ('Qn', complex_t, 8)]),
	########### Hard part #############################
	('dX_dpT_dy_pred', [(s, float_t, JEC['pred-pT'].shape[0]) 
						for (s, _) in species.get('heavy')]),
	('dX_dpT_dy_ALICE', [(s, float_t, JEC['ALICE']['Raa']['pTbins'].shape[0]) 
						for (s, _) in species.get('heavy')]),	
	('dX_dpT_dy_CMS', [(s, float_t, JEC['CMS']['Raa']['pTbins'].shape[0]) 
						for (s, _) in species.get('heavy')]),		
	('Qn_poi_pred', [(s, [('M', int_t, JEC['pred-pT'].shape[0]), 
						 ('Qn', complex_t, [JEC['pred-pT'].shape[0], 4])] )
						for (s, _) in species.get('heavy')]),	
	('Qn_ref_pred', [('M', int_t), ('Qn', complex_t, 3)]),		
	('Qn_poi_ALICE', [(s, [('M', int_t, JEC['ALICE']['vn_HF']['pTbins'].shape[0]), 
			('Qn', complex_t, [JEC['ALICE']['vn_HF']['pTbins'].shape[0], 4])] )
						for (s, _) in species.get('heavy')]),	
	('Qn_ref_ALICE', [('M', int_t), ('Qn', complex_t, 3)]),
	('Qn_poi_CMS', [(s, [('M', int_t, JEC['CMS']['vn_HF']['pTbins'].shape[0]), 
			('Qn', complex_t, [JEC['CMS']['vn_HF']['pTbins'].shape[0], 4])] )
						for (s, _) in species.get('heavy')]),	
	('Qn_ref_CMS', [('M', int_t), ('Qn', complex_t, 3)]),
]

class StopEvent(Exception):
	""" Raise to end an event early. """


def run_events(args, results_file, particles_file=None, checkpoint_ic=None):
	"""
	Run events as determined by user input:

		- Read options from `args`, as returned by `parser.parse_args()`.
		- Write results to binary file object `results_file`.
		- If `checkpoint_ic` is given, run only that IC.

	Return True if at least one event completed successfully, otherwise False.

	"""
	# set the grid step size proportionally to the nucleon width
	grid_step = .2#.15*args.nucleon_width
	# the "target" grid max: the grid shall be at least as large as the target
	grid_max_target = 15
	# next two lines set the number of grid cells and actual grid max,
	# which will be >= the target (same algorithm as trento)
	grid_n = math.ceil(2*grid_max_target/grid_step)
	grid_max = .5*grid_n*grid_step
	logging.info(
		'grid step = %.6f fm, n = %d, max = %.6f fm',
		grid_step, grid_n, grid_max
	)

	def _initial_conditions(nevents=1, initial_file='initial.hdf', avg='off'):
		"""
		Run trento and yield initial condition arrays.

		"""
		def average_ic(fname):
			with h5py.File(fname, 'a') as f:
				densityavg = np.zeros_like(f['event_0/matter_density'].value)
				Ncollavg = np.zeros_like(f['event_0/Ncoll_density'].value)
				dxy = f['event_0'].attrs['dxy']
				Neve = len(f.values())
				for eve in f.values():
					# step1, center the event
					NL = int(eve.attrs['Nx']/2)
					density = eve['matter_density'].value
					comxy = -np.array(center_of_mass(density))+np.array([NL, NL])
					density = shift(density, comxy)
					Ncoll = shift(eve['Ncoll_density'].value, comxy)
					# step2, rotate the event to align psi2
					psi2 = eve.attrs['psi2']
					imag_psi2 = psi2*180./np.pi + (90. if psi2<0 else -90.)
					densityavg += rotate(density, angle=imag_psi2, reshape=False)
					Ncollavg += rotate(Ncoll, angle=imag_psi2, reshape=False)	
				# step3 take average
				densityavg /= Neve
				Ncollavg /= Neve
			# rewrite the initial.hdf file with average ic
			with h5py.File(fname, 'w') as f:
				gp = f.create_group('avg_event')
				gp.create_dataset('matter_density', data=densityavg)
				gp.create_dataset('Ncoll_density', data=Ncollavg)
				gp.attrs.create('Nx', densityavg.shape[1])
				gp.attrs.create('Ny', densityavg.shape[0])
				gp.attrs.create('dxy', dxy)			

		try:
			os.remove(initial_file)
		except FileNotFoundError:
			pass
	   
		if avg == 'on':
			logging.info("averaged initial condition mode, could take a while")

		run_cmd(
			'trento',
			'{} {}'.format(args.proj, args.targ),
			'--number-events {}'.format(nevents if avg=='off' else 500),
			'--grid-step {} --grid-max {}'.format(grid_step, grid_max_target),
			'--output', initial_file,
			args.trento_args,
		)
		if avg == 'on':
			logging.info("taking average over 500 trento events")
			average_ic(initial_file)


		### create iterable initial conditon generator
		with h5py.File(initial_file, 'r') as f:
			for dset in f.values():
				ic = np.array(dset['matter_density'])
				nb = np.array(dset['Ncoll_density'])
				# Write the checkpoint file _before_ starting the event so that
				# even if the process is forcefully killed, the state will be
				# saved.  If / when all events complete, delete the file.
				if args.checkpoint is not None:
					with open(args.checkpoint, 'wb') as cf:
						pickle.dump((args, ic, nb), cf, pickle.HIGHEST_PROTOCOL)
					logging.info('wrote checkpoint file %s', args.checkpoint)
				yield np.array([ic, nb])


	if checkpoint_ic is None:
		# if nevents was specified, generate that number of initial conditions
		# otherwise generate indefinitely
		initial_conditions = (
			chain.from_iterable(_initial_conditions() for _ in repeat(None))
			if args.nevents is None else
			_initial_conditions(args.nevents, avg=args.avg_ic)
		)
	else:
		# just run the checkpointed IC
		initial_conditions = [checkpoint_ic[0]]
		nbinary_density = [checkpoint_ic[1]]

	# create sampler HRG object (to be reused for all events)
	hrg_kwargs = dict(species='urqmd', res_width=True)
	hrg = frzout.HRG(args.Tswitch, **hrg_kwargs)

	# append switching energy density to hydro arguments
	eswitch = hrg.energy_density()
	hydro_args = [args.hydro_args, 'edec={}'.format(eswitch)]

	# arguments for "coarse" hydro pre-runs
	# no viscosity, run down to low temperature 110 MeV
	hydro_args_coarse = [
		'etas_hrg=0 etas_min=0 etas_slope=0 zetas_max=0 zetas_width=0',
		'edec={}'.format(frzout.HRG(.110, **hrg_kwargs).energy_density())
	]

	def save_fs_with_hydro(ic):
		# roll ic by index 1 to match hydro
		#ic = np.roll(np.roll(ic, shift=-1, axis=0), shift=-1, axis=1)
		# use same grid settings as hydro output
		with h5py.File('JetData.h5','a') as f:
			taufs = f['Event'].attrs['Tau0'][0]
			dtau = f['Event'].attrs['dTau'][0]
			dxy = f['Event'].attrs['DX'][0]
			ls = f['Event'].attrs['XH'][0]
			n = 2*ls + 1
			coarse = int(dxy/grid_step+.5)
			# [tau0, tau0+dtau, tau0+2*dtau, ..., taufs - dtau] + hydro steps...
			nsteps = int(taufs/dtau)
			tau0 = taufs-dtau*nsteps
			if tau0 < 1e-2: # if tau0 too small, skip the first step
				tau0 += dtau
				nsteps -= 1
			taus = np.linspace(tau0, taufs-dtau, nsteps)
			# First, rename hydro frames and leave the first few name slots to FS
			event_gp = f['Event']
			for i in range(len(event_gp.keys()))[::-1]:
				old_name = 'Frame_{:04d}'.format(i)
				new_name = 'Frame_{:04d}'.format(i+nsteps)
				event_gp.move(old_name, new_name)
			# Second, overwrite tau0 with FS starting time, and save taufs where
			# FS and hydro is separated
			event_gp.attrs.create('Tau0', [tau0])
			event_gp.attrs.create('TauFS', [taufs])
			# Thrid, fill the first few steps with Freestreaming results
			for itau, tau in enumerate(taus):
				frame = event_gp.create_group('Frame_{:04d}'.format(itau))
				fs = freestream.FreeStreamer(ic, grid_max, tau)
				for fmt, data, arglist in [
					('e', fs.energy_density, [()]),
					('V{}', fs.flow_velocity, [(1,), (2,)]),
					('Pi{}{}', fs.shear_tensor, [(0,0), (0,1), (0,2),
														(1,1), (1,2),
															   (2,2)] ),
					]:
					for a in arglist:
						X = data(*a).T # to get the correct x-y with vishnew
						if fmt == 'V{}': # Convert u1, u2 to v1, v2
							X = X/data(0).T
						X = X[::coarse, ::coarse]
						diff = X.shape[0] - n
						start = int(abs(diff)/2)
						if diff > 0:
							# original grid is larger -> cut out middle square
							s = slice(start, start + n)
							X = X[s, s]
						elif diff < 0:
							# original grid is smaller
							#  -> create new array and place original grid in middle
							Xn = np.zeros((n, n))
							s = slice(start, start + X.shape[0])
							Xn[s, s] = X
							X = Xn
						if fmt == 'V{}':
							Comp = {1:'x', 2:'y'}
							frame.create_dataset(fmt.format(Comp[a[0]]), data=X)
						if fmt == 'e':
							frame.create_dataset(fmt.format(*a), data=X)
							frame.create_dataset('P', data=X/3.)
							frame.create_dataset('BulkPi', data=X*0.)
							prefactor = 1.0/15.62687/5.068**3 
							frame.create_dataset('Temp', data=(X*prefactor)**0.25)
							s = (X + frame['P'].value)/(frame['Temp'].value+1e-14)
							frame.create_dataset('s', data=s)
						if fmt == 'Pi{}{}': 
							frame.create_dataset(fmt.format(*a), data=X)
				pi33 = -(frame['Pi00'].value + frame['Pi11'].value \
											 + frame['Pi22'].value)
				frame.create_dataset('Pi33', data=pi33)
				pi3Z = np.zeros_like(pi33)
				frame.create_dataset('Pi03', data=pi3Z)
				frame.create_dataset('Pi13', data=pi3Z)
				frame.create_dataset('Pi23', data=pi3Z)

	def run_hydro(ic, event_size, coarse=False, dt_ratio=.25):
		"""
		Run the initial condition contained in FreeStreamer object `fs` through
		osu-hydro on a grid with approximate physical size `event_size` [fm].
		Return a dict of freeze-out surface data suitable for passing directly
		to frzout.Surface.

		Initial condition arrays are cropped or padded as necessary.

		If `coarse` is an integer > 1, use only every `coarse`th cell from the
		initial condition arrays (thus increasing the physical grid step size
		by a factor of `coarse`).  Ignore the user input `hydro_args` and
		instead run ideal hydro down to a low temperature.

		`dt_ratio` sets the timestep as a fraction of the spatial step
		(dt = dt_ratio * dxy).  The SHASTA algorithm requires dt_ratio < 1/2.

		"""
		# first freestream
		fs = freestream.FreeStreamer(ic, grid_max, args.tau_fs)
		dxy = grid_step * (coarse or 1)
		ls = math.ceil(event_size/dxy)  # the osu-hydro "ls" parameter
		n = 2*ls + 1  # actual number of grid cells
		for fmt, f, arglist in [
				('ed', fs.energy_density, [()]),
				('u{}', fs.flow_velocity, [(1,), (2,)]),
				('pi{}{}', fs.shear_tensor, [(1, 1), (1, 2), (2, 2)]),
		]:
			for a in arglist:
				X = f(*a)

				if coarse:
					X = X[::coarse, ::coarse]

				diff = X.shape[0] - n
				start = int(abs(diff)/2)

				if diff > 0:
					# original grid is larger -> cut out middle square
					s = slice(start, start + n)
					X = X[s, s]
				elif diff < 0:
					# original grid is smaller
					#  -> create new array and place original grid in middle
					Xn = np.zeros((n, n))
					s = slice(start, start + X.shape[0])
					Xn[s, s] = X
					X = Xn

				X.tofile(fmt.format(*a) + '.dat')

		dt = dxy*dt_ratio
		run_cmd(
			'osu-hydro',
			't0={} dt={} dxy={} nls={}'.format(args.tau_fs, dt, dxy, ls),
			*(hydro_args_coarse if coarse else hydro_args)
		)
		surface = np.fromfile('surface.dat', dtype='f8').reshape(-1, 26)
		# surface columns:
		#   0	 1  2  3    
		#   tau  x  y  eta  
		#   4         5         6         7
		#   dsigma_t  dsigma_x  dsigma_y  dsigma_z
		#   8    9    10
		#   v_x  v_y  v_z
		#   11    12    13    14    
		#   pitt  pitx  pity  pitz
		#         15    16    17
		#         pixx  pixy  pixz
		#               18    19
		#               piyy  piyz
		#                     20
		#                     pizz
		#   21   22   23   24   25
		#   Pi   T    e    P    muB
		if not coarse:
			logging.info("Save free streaming history with hydro histroy")
			save_fs_with_hydro(ic)

		# end event if the surface is empty -- this occurs in ultra-peripheral
		# events where the initial condition doesn't exceed Tswitch
		if surface.size == 0:
			raise StopEvent('empty surface')

		# pack surface data into a dict suitable for passing to frzout.Surface
		return dict(
				x=surface[:, 0:3],
				sigma=surface[:, 4:7],
				v=surface[:, 8:10],
				pi=dict(xx=surface.T[15],xy=surface.T[16], yy=surface.T[18]),
				Pi=surface.T[21]
			)

	results = np.empty((), dtype=result_dtype)

	def run_single_event(ic, nb, event_number):
		"""
		Run the initial condition event contained in HDF5 dataset object `ic`
		and save observables to `results`.

		"""
		results.fill(0)
		results['initial_entropy'] = ic.sum() * grid_step**2
		results['Ncoll'] = nb.sum() * grid_step**2
		logging.info("Nb %d", results['Ncoll'])
		assert all(n == grid_n for n in ic.shape)

		logging.info(
			'free streaming initial condition for %.3f fm',
			args.tau_fs
		)
		fs = freestream.FreeStreamer(ic, grid_max, args.tau_fs)

		# run coarse event on large grid and determine max radius
		rmax = math.sqrt((
			run_hydro(ic, event_size=27, coarse=3)['x'][:, 1:3]**2
		).sum(axis=1).max())
		logging.info('rmax = %.3f fm', rmax)

		# now run normal event with size set to the max radius
		# and create sampler surface object
		surface = frzout.Surface(**run_hydro(ic, event_size=rmax), ymax=2)
		logging.info('%d freeze-out cells', len(surface))

		# Sampling particle for UrQMD events
		logging.info('sampling surface with frzout')
		minsamples, maxsamples = 1, 10  # reasonable range for nsamples
		minparts = 10**5  # min number of particles to sample
		nparts = 0  # for tracking total number of sampled particles
		with open('particles_in.dat', 'w') as f:
			for nsamples in range(1, maxsamples + 1):
				parts = frzout.sample(surface, hrg)
				if parts.size == 0:
					continue
				nparts += parts.size
				print('#', parts.size, file=f)
				for p in parts:
					print(p['ID'], *p['x'], *p['p'], file=f)
				if nparts >= minparts and nsamples >= minsamples:
					break

		results['nsamples'] = nsamples
		logging.info('produced %d particles in %d samples', nparts, nsamples)

		if nparts == 0:
			raise StopEvent('no particles produced')

		# ==================Heavy Flavor===========================
		# Run Pythia+Lido
		prefix = os.environ.get('XDG_DATA_HOME')
		cmd = "hydro-couple {:s}/pythia-setting.txt initial.hdf {:d} JetData.h5 {:s}/settings.xml {:d} {:f} {:f} {:f} {:f}"
		run_cmd(cmd.format(prefix, event_number-1, prefix, args.NPythiaEvents,
						   1.0, -1, 0.6, 0.0))

		# hadronization
		hq = 'c'
		prefix = os.environ.get('XDG_DATA_HOME')+"/hvq-hadronization/"
		os.environ["ftn20"] = "{}-meson-frzout.dat".format(hq)
		os.environ["ftn30"] = prefix+"parameters_{}_hd.dat".format(hq)
		os.environ["ftn40"] = prefix+"recomb_{}_tot.dat".format(hq)
		os.environ["ftn50"] = prefix+"recomb_{}_BR1.dat".format(hq)
		logging.info(os.environ["ftn30"])
		subprocess.run("hvq-hadronization", stdin=open("{}-quark-frzout.dat".format(hq)))

		# ==================Heavy + Soft --> UrQMD===========================
		run_cmd('run-urqmd {}'.format(nsamples) )

		# read final particle data
		ID, charge, fmass, px, py, pz, y, eta, pT0, y0, w, _ = (
			np.array(col, dtype=dtype) for (col, dtype) in
			zip(
				zip(*read_text_file('particles_out.dat')),
				(2*[int] + 10*[float])
			)
		)
		# pT, phi, and id cut
		pT = np.sqrt(px**2+py**2)
		phi = np.arctan2(py, px)
		charged = (charge != 0)
		abs_eta = np.fabs(eta)
		abs_ID = np.abs(ID)
		# It may be redunant to find b-hadron at this stage since UrQMD has
		# not included them yet
		heavy_pid = [pid for (_, pid) in species.get('heavy')]
		is_heavy = np.array([u in heavy_pid for u in abs_ID], dtype=bool)
		is_light = np.logical_not(is_heavy)

		#============for soft particles======================
		results['dNch_deta'] = \
					np.count_nonzero(charged & (abs_eta<.5) & is_light) / nsamples

		for exp in ['ALICE', 'CMS']:
			#=========Event plane Q-vector from UrQMD events======================
			phi_light = phi[charged & is_light \
				& (JEC[exp]['vn_ref']['ybins'][0] < eta) \
				& (eta < JEC[exp]['vn_ref']['ybins'][1]) \
				& (JEC[exp]['vn_ref']['pTbins'][0] < pT) \
				& (pT < JEC[exp]['vn_ref']['pTbins'][1])]
			results['Qn_ref_'+exp]['M'] = phi_light.shape[0]
			results['Qn_ref_'+exp]['Qn'] = np.array([np.exp(1j*n*phi_light).sum() 
											for n in range(1, 4)])
			#===========For heavy particles======================
			# For charmed hadrons, use info after urqmd
			HF_dict = { 'pid': abs_ID[is_heavy],
						'pT' : pT[is_heavy],
						'y'  : y[is_heavy],
						'phi': phi[is_heavy],
						'w' : w[is_heavy] # normalized to an area units
			  	}
			POI = [pid for (_, pid) in species.get('heavy')]
			flow = JLP.Qvector(HF_dict, JEC[exp]['vn_HF']['pTbins'],
								JEC[exp]['vn_HF']['ybins'], POI, order=4)
			Yield = JLP.Yield(HF_dict, JEC[exp]['Raa']['pTbins'],
								JEC[exp]['Raa']['ybins'], POI)
			for (s, pid) in species.get('heavy'):
				results['dX_dpT_dy_'+exp][s] = Yield[pid][:,0]
				results['Qn_poi_'+exp][s]['M'] = flow[pid]['M'][:,0]
				results['Qn_poi_'+exp][s]['Qn'] = flow[pid]['Qn'][:,0,:]

		# For full pT prediction
		#=========Use high precision Q-vector at the end of hydro==============
		# oversample to get a high percision event plane at freezeout
		ophi_light = np.empty(0)
		nloop=0
		while ophi_light.size < 10**6 and nloop < 100000:
			nloop += 1
			oE, opx, opy, opz = frzout.sample(surface, hrg)['p'].T
			oM, opT, oy, ophi = JLP.fourvec_to_curvelinear(opx, opy, opz, oE)
			ophi = ophi[(-2 < oy) & (oy < 2) & (0.2 < opT) & (opT <5.0)]
			ophi_light = np.append(ophi_light, ophi)
		results['Qn_ref_pred']['M'] = ophi_light.shape[0]
		results['Qn_ref_pred']['Qn'] = np.array([np.exp(1j*n*ophi_light).sum() 
											for n in range(1, 4)])
		del ophi_light
		#===========For heavy particles======================
		# For charmed hadrons, use info after urqmd
		HF_dict = { 'pid': abs_ID[is_heavy],
					'pT' : pT[is_heavy],
					'y'  : y[is_heavy],
					'phi': phi[is_heavy],
					'w' : w[is_heavy]
		  		}
		POI = [pid for (_, pid) in species.get('heavy')]
		flow = JLP.Qvector(HF_dict, JEC['pred-pT'], [[-2,2]], POI, order=4)
		Yield = JLP.Yield(HF_dict, JEC['pred-pT'], [[-1,1]], POI)
		for (s, pid) in species.get('heavy'):
			results['dX_dpT_dy_pred'][s] = Yield[pid][:,0]
			results['Qn_poi_pred'][s]['M'] = flow[pid]['M'][:,0]
			results['Qn_poi_pred'][s]['Qn'] = flow[pid]['Qn'][:,0]
		POI = [pid for (_, pid) in species.get('heavy')]
		#^^^^^^end of run_single_event(...)^^^^^^^^^^^^^^^^^^ 
	nfail = 0

	# run each initial condition event and save results to file
	for n, (ic, nb) in enumerate(initial_conditions, start=1):
		logging.info('starting event %d', n)

		try:
			run_single_event(ic, nb, n)
		except StopEvent as e:
			if particles_file is not None:
				particles_file.create_dataset(
					'event_{}'.format(n), shape=(0,), dtype=parts_dtype
				)
			logging.info('event stopped: %s', e)
		except Exception:
			logging.exception('event %d failed', n)
			nfail += 1
			if nfail > 3 and nfail/n > .5:
				logging.critical('too many failures, stopping events')
				break
			logging.warning('continuing to next event')
			continue

		results_file.write(results.tobytes())
		logging.info('event %d completed successfully', n)

	# end of events: if running with a checkpoint, delete the file unless this
	# was a failed re-run of a checkpoint event
	if args.checkpoint is not None:
		if checkpoint_ic is not None and nfail > 0:
			logging.info(
				'checkpoint event failed, keeping file %s',
				args.checkpoint
			)
		else:
			os.remove(args.checkpoint)
			logging.info('removed checkpoint file %s', args.checkpoint)

	return n > nfail


def main():
	args, checkpoint_ic = parse_args_checkpoint()

	if checkpoint_ic is None:
		# starting fresh -> truncate output files
		filemode = 'w'

		# must handle rank first since it affects paths
		if args.rankvar:
			rank = os.getenv(args.rankvar)
			if rank is None:
				sys.exit('rank variable {} is not set'.format(args.rankvar))

			if args.rankfmt:
				rank = args.rankfmt.format(int(rank))

			# append rank to path arguments, e.g.:
			#   /path/to/output.log  ->  /path/to/output/<rank>.log
			for a in ['results', 'logfile', 'particles', 'checkpoint']:
				value = getattr(args, a)
				if value is not None:
					root, ext = os.path.splitext(value)
					setattr(args, a, os.path.join(root, rank) + ext)
	else:
		# running checkpoint event -> append to existing files
		filemode = 'a'

	os.makedirs(os.path.dirname(args.results), exist_ok=True)

	if args.logfile is None:
		logfile_kwargs = dict(stream=sys.stdout)
	else:
		logfile_kwargs = dict(filename=args.logfile, filemode=filemode)
		os.makedirs(os.path.dirname(args.logfile), exist_ok=True)

	if args.particles is not None:
		os.makedirs(os.path.dirname(args.particles), exist_ok=True)

	if args.checkpoint is not None:
		os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

	logging.basicConfig(
		level=getattr(logging, args.loglevel.upper()),
		format='[%(levelname)s@%(relativeCreated)d] %(message)s',
		**logfile_kwargs
	)
	logging.captureWarnings(True)

	start = datetime.datetime.now()
	if checkpoint_ic is None:
		logging.info('started at %s', start)
		logging.info('arguments: %r', args)
	else:
		logging.info(
			'restarting from checkpoint file %s at %s',
			args.checkpoint, start
		)

	# translate SIGTERM to KeyboardInterrupt
	signal.signal(signal.SIGTERM, signal.default_int_handler)
	logging.debug('set SIGTERM handler')

	@contextmanager
	def h5py_file():
		yield h5py.File(args.particles, 'w') if args.particles else None

	with \
			open(args.results, filemode + 'b',
				 buffering=args.buffering) as results_file, \
			h5py_file() as particles_file, \
			tempfile.TemporaryDirectory(
				prefix='hic-', dir=args.tmpdir) as workdir:
		os.chdir(workdir)
		logging.info('working directory: %s', workdir)

		try:
			status = run_events(args, results_file, particles_file, checkpoint_ic)
		except KeyboardInterrupt:
			# after catching the initial SIGTERM or interrupt, ignore them
			# during shutdown -- this ensures everything will exit gracefully
			# in case of additional signals (short of SIGKILL)
			signal.signal(signal.SIGTERM, signal.SIG_IGN)
			signal.signal(signal.SIGINT, signal.SIG_IGN)
			status = True
			logging.info(
				'interrupt or signal at %s, cleaning up...',
				datetime.datetime.now()
			)
			if args.checkpoint is not None:
				logging.info(
					'current event saved in checkpoint file %s',
					args.checkpoint
				)

	end = datetime.datetime.now()
	logging.info('finished at %s, %s elapsed', end, end - start)

	if not status:
		sys.exit(1)


if __name__ == "__main__":
	main()

