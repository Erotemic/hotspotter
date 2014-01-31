from hscom import helpers
from hotspotter import nn_filters


exec(open(nn_filters.__file__).read())
qcx2_nns = helpers.load_testdata('qcx2_nns')
