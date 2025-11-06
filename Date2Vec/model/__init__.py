import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../model'))
#from model.DeepTD_LSP import *
# from model.DeepTD_simplized_patch_V1 import *
# from model.Transformer_D2V import *
# from model.DeepTD_patch_koopa_V2 import *
# from model.DeepTD_simplized import *
# from model.D2V_Attention import *
# from model.D2V_Linear import *
# from model.T2V_Fusion import *
# from model.posisition_Fusion import *


# 消融测试

from model.Position_Transformer import *
from model.D2V_Fourier_Transformer import *
from model.D2V_Fourier_PatchTST2 import *
from model.D2V_Fourier_ITransformer import *
from model.D2V_Fourier_Autoformer import D2V_Autoformer
from model.D2V_Fourier_Fedformer import D2V_Fedformer
from model.GLAFF_ITransformer import GLAFF_iTransformer
from model.GLAFF_PatchTST import GLAFF_PatchTST
from model.GLAFF_Transformer import GLAFF_Transformer
from model.T2V_Transformer import T2V_Transformer
from model.T2V_ITransformer import T2V_iTransformer
from model.T2V_PatchTST import T2V_PatchTST
from model.Transformer import Transformer
from model.PatchTST import PatchTST
from model.ITransformer import iTransformer