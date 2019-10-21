from model import *



path = "/research/byu2/hgeng/metric-learning/vias/test/Nvia69822_mb_mb_lccout.oas.gds.png"


feature = get_feature(path, 100, 20, 32)

print(feature.shape)



