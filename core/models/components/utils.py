# Project:
#   Localized Questions in VQA
# Description:
#   Utilities for VQA components
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

def expand_like_2D(to_expand, reference):
    # expands input with dims [B, K] to dimensions of reference which are [B, K, M, M]
    expanded = to_expand.unsqueeze_(2).unsqueeze_(3).expand_as(reference)
    return expanded