"""Import the actual UP registry after explicit platform configuration."""
import argparse
import json
from up_platform import configure_platform

parser = argparse.ArgumentParser()
parser.add_argument('--vendor', required=True)
parser.add_argument('--purpose', default='training')
args = parser.parse_args()
platform = configure_platform({'backend': args.vendor}, purpose=args.purpose)
import up
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY, RUNNER_REGISTRY
print(json.dumps({'up_version': up.__version__, 'vendor': platform.vendor,
                  'models': list(MODULE_ZOO_REGISTRY), 'runners': list(RUNNER_REGISTRY)}))
