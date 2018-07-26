import os
import dask
import shutil
from pymir import mirstr as m


class uv():
    """A class instance to manage imaging a single UV file, assuming
    CABB data and a MFCLEAN style image deconvolution. At least 
    initially
    """
    def __init__(self, uv: str):
        """Initialise the container.
        NOTE: This signature may change. Will also have to be
              updated in self.selfcal()
        .
        Arguments:
            uv {str} -- The name of the uv-file to image
        """
        if not os.path.exists(uv):
            raise OSError(f"File {uv} not found")

        self.uv = uv
        self.img_tasks = {}

        
    def __str__(self):
        """Neat representation to print
        """
        return f"uv({self.uv})"


    @property
    def resolution(self):
        """Fit to the restor image to return the image resolutions
        """
        if 'restor' not in self.img_tasks.keys():
            return None

        invert = self.img_tasks['invert']
        imfit = m(f"imfit in={invert.beam} object=beam region='perc(1)(1)'").run()
        print(imfit)

        src_data = {}
        obj1 = False
        for line in imfit.p.stdout.split('\n'):
            if 'Object type: beam' in line:
                obj1 = True
                continue
            elif not obj1:
                continue
            
            if 'Major axis' in line:
                items = line.split()
                src_data['bmaj'] = float(items[-3])
                src_data['bmaj_e'] = float(items[-1])
            elif 'Minor axis' in line:
                items = line.split()
                src_data['bmin'] = float(items[-3])
                src_data['bmin_e'] = float(items[-1])
            elif 'Position angle' in line:
                items = line.split()
                src_data['pa'] = float(items[-3])
                src_data['pa_e'] = float(items[-1])            

        return src_data


    def image(self, invert_kwargs: dict=None):
        """A simple MFCLEAN imaging pipeline
        
        Keyword Arguments:
            invert_kwargs {dict} -- Options to pass to the invert map/beam stage (default: {None})
        """

        # No work to do, as data has been imaged
        if 'restor' in self.img_tasks.keys():
            return

        print(self.uv)
        invert = m(f"invert vis={self.uv} options=mfs,sdb,double,mosaic " \
                   f"offset=3:32:22.0,-27:48:37 stokes=i imsize=4,4,beam " \
                   f"map={self.uv}.map beam={self.uv}.beam robust=2 cell=0.35", 
                    over=invert_kwargs).run()
        print(invert)

        stokes_v =  m(f"invert vis={self.uv} imsize=3,3,beam options=mfs,sdb,double,mosaic " \
                      f"offset=3:32:22.0,-27:48:37 stokes=v imsize=2,2,beam " \
                      f"map={self.uv}.v.map cell=0.35").run()
        print(stokes_v)

        sigest = m(f"sigest in={stokes_v.map}").run()
        print(sigest)
        for l in sigest.p.stdout.split('\n'):
            if 'Estimated rms' in l:
                v_rms = float(l.split()[-1])

        mfclean = m(f"mfclean map={invert.map} beam={invert.beam} out={self.uv}.clean "\
                    f"region='perc(66)' niters=1500 cutoff={5*v_rms}").run()
        print(mfclean)

        restor = m(f"restor map={invert.map} beam={invert.beam} model={mfclean.out} "\
                   f"options=mfs out={self.uv}.restor").run()
        print(restor)

        self.img_tasks = {'invert':invert, 'invert_v':stokes_v, 'mfclean':mfclean,
                          'restor':restor, 'stokes_v_rms':v_rms}


    def convol(self, bmaj: float, bmin: float, pa: float):
        """Convol a restored image to a different resolution
        
        Arguments:
            bmaj {float} -- Beam major in arcseconds
            bmin {float} -- Beam minor in arcseconds
            pa {float} -- Beam position angle in degrees
        """

        if 'restor' not in self.img_tasks.keys() or \
                    'convol' in self.img_tasks.keys():
            return None

        restor = self.img_tasks['restor']
        convol = m(f"convol map={restor.out} out={self.uv}.convol fwhm={bmaj},{bmin} "
                   f"pa={pa} options=final").run()
        print(convol)

        self.img_tasks['convol'] = convol


    def attempt_selfcal(self, mode: str=None):
        """Logic to decide whether selcalibration should be attempted. Mode can be
        a function that is passed into the method. It should except as a single 
        argument a reference to uv-instance. 

        Keyword Arguments:
            mode {str} -- The mode used to evaluate decision to self (default: {None})
        
        Returns:
            bool -- Whether selfcal should be used
            dict -- optional return with selfcal key/values to use
        """

        # Nothing set
        if mode is None:
            return True

        # If it quacks like a duck
        try:
            return mode(self)
        except:
            pass

        if mode == 'test':
            # Example of conditional. Used to test subequent imaging
            if '100' in self.uv:
                return False

        elif mode == 'restor_max':
            from astropy.io import fits as pyfits
            restor = self.img_tasks['restor']
            fits = m(f"fits in={restor.out} out={restor.out}.fits op=xyout " \
                      "region='images(1,1)'").run()
            data = pyfits.open(fits.out)[0].data.squeeze()
            delete_miriad(fits.out)

            # Threshold selected by dumb luck and subsequet experimentation
            restor_max = data.max()
            if restor_max > 150*self.img_tasks['stokes_v_rms']:
                return True, {'options':'mfs,amp'}
            elif restor_max > 50*self.img_tasks['stokes_v_rms']:
                return True
            else:
                return False

        elif mode == 'clean_sum':
            from astropy.io import fits as pyfits
            clean = self.img_tasks['mfclean']
            fits = m(f"fits in={clean.out} out={clean.out}.fits op=xyout " \
                      "region='images(1,1)'").run()
            data = pyfits.open(fits.out)[0].data.squeeze()
            delete_miriad(fits.out)

            clean_sum = data.sum()
            if clean_sum > 500*self.img_tasks['stokes_v_rms']:
                return True, {'options':'mfs,amp'}
            elif clean_sum > 100*self.img_tasks['stokes_v_rms']:
                return True
            else:
                return False 

        return True


    def selfcal(self, *args, round: int=0, self_kwargs: dict=None, **kwargs):
        """Apply a round of selfcalibration to the uv-file if appropriate.
        
        Keyword Arguments:
            round {int} -- The selfcalibration round the process is up to (default: {0})
            self_kwargs {dict} -- Options for selfcalibration. Can be overwritten by attempt_selfcal returns (default: {None})
        """
        run_self = self.attempt_selfcal(**kwargs)
        
        if isinstance(run_self, tuple):
            run_self, self_kwargs = run_self

        # Conditions are not met. Return instance of self.
        if not run_self:
            return self

        # Apply calibration tables in existing uv-file
        uvaver = m(f"uvaver vis={self.uv} out={self.uv}.{round}").run()
        print(uvaver)

        # Derive selfcalibrated solutions
        # TODO: Discuss changes to interval and nfbins, especially with two IFS
        # TODO: in the uvcat'ed uvfiles. 
        selfcal = m(f"selfcal vis={uvaver.out} model={self.img_tasks['mfclean'].out} "\
                    f"options=mfs,phase interval=0.1 nfbin=4", over=self_kwargs).run()
        print(selfcal)

        # Return new instance to selfcaled file
        return uv(uvaver.out)


def delete_miriad(f:str):
    """Delete a folder if it exists
    
    Arguments:
        f {str} -- Folder to delete
    """
    if os.path.exists(f):
        if os.path.isdir(f):
            print(f"Removing directory {f}")
            shutil.rmtree(f)
        else:
            print(f"Removing file {f}")
            os.remove(f)

@dask.delayed
def run_linmos(mos: list, round: int=0):
    """The the miriad task linmos agaisnt list of uv files
    
    Arguments:
        mos {list} -- A list of uv-instances
        round {int} -- A int for self calibration round
    """
    convol_files = f'temp_file_{round}.dat'
    with open(f"{convol_files}", 'w') as f:
        for i in mos:
            print(f"{i.img_tasks['convol'].out}", file=f)

    out = f"all_days.{round}.linmos"
    delete_miriad(out)
    linmos = m(f"linmos in=@{convol_files} bw=4.096 out={out}").run()
    print(linmos)

    os.remove(convol_files)

    return linmos


@dask.delayed
def run_image(s: str, invert_kwargs: dict= None):
    """Function to Dask-ify for distribution
    
    Arguments:
        s {str} -- path to uv-file
        invert_kwargs {dict} -- invert options
    """
    if isinstance(s, str):
        point = uv(s)
    else:
        point = s

    point.image(invert_kwargs=invert_kwargs)
    point.convol(7., 3., 2.2)    

    return point


@dask.delayed
def run_selfcal(s: uv, round: int, *args, **kwargs):
    """Run the attempt_selfcal method
    
    Arguments:
        s {uv} -- uv file to self calibrate
        round {int} -- the selfcalibration round
    """
    return s.selfcal(round=round, **kwargs)

@dask.delayed
def dask_reduce(arr):
    """Reduce the worker graph
    
    Arguments:
        arr {list} -- Reduce the Dask graph
    """
    return arr

# ------------------------------------------------------------------------------------
def sc_round_1(s):
    """Logic for the first round of selfcalibration
    
    Arguments:
        s {uv} -- An instance of the uv-class. 
    """
    from astropy.io import fits as pyfits
    restor = s.img_tasks['restor']
    fits = m(f"fits in={restor.out} out={restor.out}.fits op=xyout " \
                "region='images(1,1)'").run()
    data = pyfits.open(fits.out)[0].data.squeeze()
    delete_miriad(fits.out)

    # Threshold selected by dumb luck and subsequet experimentation
    restor_max = data.max()
    if restor_max > 50*s.img_tasks['stokes_v_rms']:
        return True, {'interval':'0.5'}
    else:
        return False

def sc_round_2(s):
    """Logic for the second round of selfcalibration
    
    Arguments:
        s {uv} -- An instance of the uv-class. 
    """
    from astropy.io import fits as pyfits
    restor = s.img_tasks['restor']
    fits = m(f"fits in={restor.out} out={restor.out}.fits op=xyout " \
                "region='images(1,1)'").run()
    data = pyfits.open(fits.out)[0].data.squeeze()
    delete_miriad(fits.out)

    # Threshold selected by dumb luck and subsequet experimentation
    restor_max = data.max()
    if restor_max > 50*s.img_tasks['stokes_v_rms']:
        return True
    else:
        return False

def sc_round_3(s):
    """Logic for the third round of selfcalibration
    
    Arguments:
        s {uv} -- An instance of the uv-class. 
    """
    from astropy.io import fits as pyfits
    restor = s.img_tasks['restor']
    fits = m(f"fits in={restor.out} out={restor.out}.fits op=xyout " \
                "region='images(1,1)'").run()
    data = pyfits.open(fits.out)[0].data.squeeze()
    delete_miriad(fits.out)

    # Threshold selected by dumb luck and subsequet experimentation
    restor_max = data.max()
    if restor_max > 150*s.img_tasks['stokes_v_rms']:
        return True, {'options':'mfs,amp', 'interval':'0.5'}
    else:
        return False


# ------------------------------------------------------------------------------------

if __name__ == '__main__':
    files = [ 'c3171_95.uv', 'c3171_96.uv',  'c3171_97.uv',  'c3171_98.uv',
              'c3171_99.uv', 'c3171_100.uv','c3171_101.uv', 'c3171_102.uv',
             'c3171_103.uv', 'c3171_104.uv','c3171_105.uv', 'c3171_106.uv']

    from glob import glob
    files = glob("*.uv")

    # clean up existing images for testing
    for test in files:
        for c in [0, 1]:
            for i in ['map', 'beam', 'clean', 'restor', 'v.map', 'convol', 'restor.fits']:
                if c == 0:
                    f = f"{test}.{i}"
                else:
                    delete_miriad(f"{test}.{c}")
                    f = f"{test}.{c}.{i}"
                delete_miriad(f)

    linmos_imgs = []

    # Example code to get to run with Dask framework
    imgs = [run_image(f, invert_kwargs={'imsize':'4,4,beam'}) for f in files]
    e1 = run_linmos(imgs, 0)
    linmos_imgs.append(e1)

    self_imgs = [run_selfcal(uv, 1, mode=sc_round_1) for uv in imgs]
    self_imgs = [run_image(uv) for uv in self_imgs]
    e2 = run_linmos(self_imgs, 1)
    linmos_imgs.append(e2)

    self_imgs1 = [run_selfcal(uv, 2, mode=sc_round_2) for uv in self_imgs]
    self_imgs1 = [run_image(uv) for uv in self_imgs1]
    e3 = run_linmos(self_imgs1, 2)
    linmos_imgs.append(e3)

    self_imgs2 = [run_selfcal(uv, 3, mode=sc_round_3) for uv in self_imgs1]
    self_imgs2 = [run_image(uv) for uv in self_imgs2]
    e4 = run_linmos(self_imgs1, 3)
    linmos_imgs.append(e4)

    dask_reduce(linmos_imgs).visualize('graph.png')
    dask_reduce(linmos_imgs).compute()

    # import pickle
    # print(pickle.dumps(c100))