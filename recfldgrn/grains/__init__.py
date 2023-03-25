Field2Grain_2_get_GrnStrList_Fn = {}
Field2Grain_2_get_GrnIdxList_Fn = {}
Field2Grain_2_get_GrnWgtList_Fn = {}


## ------------------------------------(categorical)------------------------------------
###### P Rec
fld2grn = 'P@basicInfo-basicInfoDftGrn'
from .cate.ptdemo_grn import get_basicInfoGrn_list
Field2Grain_2_get_GrnStrList_Fn[fld2grn] = get_basicInfoGrn_list
Field2Grain_2_get_GrnIdxList_Fn[fld2grn] = lambda x, v2idx: [v2idx.get(i) for i in get_basicInfoGrn_list(x)]

###### EC Rec
fld2grn = 'EC@BasicInfo-BasicDftGrn'
from .cate.ecbasic_grn import get_basicInfoGrn_list
Field2Grain_2_get_GrnStrList_Fn[fld2grn] = get_basicInfoGrn_list
Field2Grain_2_get_GrnIdxList_Fn[fld2grn] = lambda x, v2idx: [v2idx.get(i) for i in get_basicInfoGrn_list(x)]

###### Diag Rec
fld2grn = 'Diag@Value-DiagDftGrn'
from .cate.diag_grn import func_convert_DiagVgrn
Field2Grain_2_get_GrnStrList_Fn[fld2grn] = func_convert_DiagVgrn
Field2Grain_2_get_GrnIdxList_Fn[fld2grn] = lambda x, v2idx: [v2idx.get(i) for i in func_convert_DiagVgrn(x)]

###### Smoking Rec
fld2grn = 'Smoking@V-SmokingDftGrn'
from .cate.smoking_grn import func_convert_Smokinggrn
Field2Grain_2_get_GrnStrList_Fn[fld2grn] = func_convert_Smokinggrn
Field2Grain_2_get_GrnIdxList_Fn[fld2grn] = lambda x, v2idx: [v2idx.get(i) for i in func_convert_Smokinggrn(x)]

###### PNSect Rec
fld2grn = 'PNSect@SectName-PNSctNmDftGrn'
from .cate.pnsectnm_grn import get_SectNmDftGrn_list
Field2Grain_2_get_GrnStrList_Fn[fld2grn] = get_SectNmDftGrn_list
Field2Grain_2_get_GrnIdxList_Fn[fld2grn] = lambda x, v2idx: [v2idx.get(i) for i in get_SectNmDftGrn_list(x)]


## ------------------------------------(datetime)------------------------------------
###### EC Rec
fld2grn = 'EC@DT_min-DTDftGrn'
from .misc.dt_grn import func_convert_DTgrn
Field2Grain_2_get_GrnStrList_Fn[fld2grn] = func_convert_DTgrn
Field2Grain_2_get_GrnIdxList_Fn[fld2grn] = lambda x, v2idx: [v2idx.get(i) for i in func_convert_DTgrn(x)]

###### Diag Rec
fld2grn = 'Diag@DT-DTDftGrn'
from .misc.dt_grn import func_convert_DTgrn
Field2Grain_2_get_GrnStrList_Fn[fld2grn] = func_convert_DTgrn
Field2Grain_2_get_GrnIdxList_Fn[fld2grn] = lambda x, v2idx: [v2idx.get(i) for i in func_convert_DTgrn(x)]

###### A1C Rec
fld2grn = 'A1C@DT-DTDftGrn'
from .misc.dt_grn import func_convert_DTgrn
Field2Grain_2_get_GrnStrList_Fn[fld2grn] = func_convert_DTgrn
Field2Grain_2_get_GrnIdxList_Fn[fld2grn] = lambda x, v2idx: [v2idx.get(i) for i in func_convert_DTgrn(x)]

###### Smoking Rec
fld2grn = 'Smoking@DT-DTDftGrn'
from .misc.dt_grn import func_convert_DTgrn
Field2Grain_2_get_GrnStrList_Fn[fld2grn] = func_convert_DTgrn
Field2Grain_2_get_GrnIdxList_Fn[fld2grn] = lambda x, v2idx: [v2idx.get(i) for i in func_convert_DTgrn(x)]

###### PN Rec
fld2grn = 'PN@DT-DTDftGrn'
from .misc.dt_grn import func_convert_DTgrn
Field2Grain_2_get_GrnStrList_Fn[fld2grn] = func_convert_DTgrn
Field2Grain_2_get_GrnIdxList_Fn[fld2grn] = lambda x, v2idx: [v2idx.get(i) for i in func_convert_DTgrn(x)]


## ------------------------------------(numeric)------------------------------------
###### P Rec
fld2grn = 'P@age-AgeNumeDftGrn'
from .nume.age_grn import func_convert_AGEwgt
Field2Grain_2_get_GrnIdxList_Fn[fld2grn] = lambda x, v2idx: list(range(len(v2idx)))
Field2Grain_2_get_GrnWgtList_Fn[fld2grn] = func_convert_AGEwgt

###### A1C Rec
fld2grn = 'A1C@V-A1CNumeDftGrn'
from .nume.a1c_grn import func_convert_A1Cwgt
Field2Grain_2_get_GrnIdxList_Fn[fld2grn] = lambda x, v2idx: list(range(len(v2idx)))
Field2Grain_2_get_GrnWgtList_Fn[fld2grn] = func_convert_A1Cwgt


## ------------------------------------(textual)------------------------------------
###### PNSectSent Rec
fld2grn = 'PNSectSent@Sentence-Tk@TknzLLMGrn'
from .text.tkllm_grn import func_convert_SentTkLLM
Field2Grain_2_get_GrnIdxList_Fn[fld2grn] = lambda x, tokenizer: func_convert_SentTkLLM(x, tokenizer)

