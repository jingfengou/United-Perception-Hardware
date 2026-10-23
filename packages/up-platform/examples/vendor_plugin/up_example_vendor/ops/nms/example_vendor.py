"""Only one operation is overridden; every other compatible operation is shared."""


def naive_nms(dets, thresh, offset=0):
    # Replace this body with the vendor's full, contract-compatible implementation.
    from up_platform.ops.nms.common import naive_nms as shared_nms
    return shared_nms(dets, thresh, offset=offset)
