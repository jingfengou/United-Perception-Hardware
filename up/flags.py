"""Shared configuration flags without importing application packages."""


class QuantFlag:
    def __init__(self):
        self.flag = False


class DeployFlag(QuantFlag):
    pass


class Aligned:
    def __init__(self):
        self.aligned = False
        self.offset = 1


class Fp16Flag:
    def __init__(self):
        self.fp16 = False


class IterBase(QuantFlag):
    pass


class Mosaic_p:
    def __init__(self):
        self.flag = True


class DistBackend:
    def __init__(self):
        self.backend = "dist"


ITER_BASE_FLAG = IterBase()
FP16_FLAG = Fp16Flag()
ALIGNED_FLAG = Aligned()
DIST_BACKEND = DistBackend()
QUANT_FLAG = QuantFlag()
DEPLOY_FLAG = QuantFlag()
