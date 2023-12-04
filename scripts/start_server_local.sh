set -ex
cd ../pokemon-showdown
cp ../data/showdown-config-local.js ../pokemon-showdown/config/config.js
node pokemon-showdown start --no-security
