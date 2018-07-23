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
        """Initialise the container
        
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


    def image(self):
        """Create the miriad images
        """

        invert = m(f"invert vis={self.uv} imsize=3,3,beam options=mfs,sdb,double,mosaic " \
                   f"offset=3:32:22.0,-27:48:37 stokes=i imsize=5,5,beam " \
                   f"map={self.uv}.map beam={self.uv}.beam").run()
        print(invert)

        stokes_v =  m(f"invert vis={self.uv} imsize=3,3,beam options=mfs,sdb,double,mosaic " \
                      f"offset=3:32:22.0,-27:48:37 stokes=v imsize=3,3,beam " \
                      f"map={self.uv}.v.map").run()
        print(stokes_v)

        sigest = m(f"sigest in={stokes_v.map}").run()
        print(sigest)
        for l in sigest.p.stdout.split('\n'):
            if 'Estimated rms' in l:
                v_rms = float(l.split()[-1])

        mfclean = m(f"mfclean map={invert.map} beam={invert.beam} out={self.uv}.clean "\
                    f"region='perc(66)' niters=500 cutoff={5*v_rms}").run()
        print(mfclean)

        restor = m(f"restor map={invert.map} beam={invert.beam} model={mfclean.out} "\
                   f"options=mfs out={self.uv}.restor").run()
        print(restor)

        self.img_tasks = {'invert':invert, 'invert_v':stokes_v, 'mfclean':mfclean,
                          'restor':restor}


    def convol(self, bmaj: float, bmin: float, pa: float):
        """Convol a restored image to a different resolution
        
        Arguments:
            bmaj {float} -- Beam major in arcseconds
            bmin {float} -- Beam minor in arcseconds
            pa {float} -- Beam position angle in degrees
        """
        if 'restor' not in self.img_tasks.keys():
            return None

        restor = self.img_tasks['restor']
        convol = m(f"convol map={restor.out} out={self.uv}.convol fwhm={bmaj},{bmin} "
                   f"pa={pa} options=final").run()
        print(convol)

        self.img_tasks['convol'] = convol


def delete_miriad(f:str):
    """Delete a folder if it exists
    
    Arguments:
        f {str} -- Folder to delete
    """
    if os.path.exists(f):
        print(f"Removing {f}")
        shutil.rmtree(f)


@dask.delayed
def run_linmos(imgs: list, round:int =0):
    """The the miriad task linmos agaisnt list of uv files
    
    Arguments:
        imgs {list} -- A list of uv-instances
        round {int} -- A int for self calibration round
    """
    convol_files = f'temp_file_{round}.dat'
    with open(f"{convol_files}", 'w') as f:
        for i in imgs:
            print(f"{i.img_tasks['convol'].out}", file=f)

    out = f"all_days.{round}.linmos"
    delete_miriad(out)
    linmos = m(f"linmos in=@{convol_files} bw=4.096 out={out}").run()
    print(linmos)

    os.remove(convol_files)

    return linmos


@dask.delayed
def run_image(s: str):
    """Function to Dask-ify for distribution
    
    Arguments:
        s {str} -- path to uv-file
    """
    point = uv(s)
    point.image()
    point.convol(7., 3., 2.2)    

    return point

if __name__ == '__main__':
    files = ['c3171_99.uv', 'c3171_100.uv','c3171_101.uv', 'c3171_102.uv']

    # clean up existing images for testing
    for test in files:
        for i in ['map', 'beam', 'clean', 'restor', 'v.map', 'convol']:
            f = f"{test}.{i}"
            delete_miriad(f)

    imgs = [run_image(f) for f in files]
    run_linmos(imgs).compute()

    # import pickle
    # print(pickle.dumps(c100))