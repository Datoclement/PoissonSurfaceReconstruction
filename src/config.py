import os

####################
# file system misc #
####################

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_dir = 'local'
input_dir = 'data'

inpdir = os.path.join(root, input_dir)
outdir = os.path.join(root, output_dir)


inpfile = 'bunny_normals'
outsufx = 'last_test'
sufx = '.ply'

inppath = os.path.join(inpdir, inpfile + sufx)
outpath = os.path.join(outdir, '_'.join([inpfile, outsufx]) + sufx)

#############################
# algorithm hyperparameters #
#############################

octdepth = 3
divtempt = 1.

def test():
    print(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    test()
