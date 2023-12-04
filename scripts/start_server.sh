set -ex
cd ../pokemon-showdown
cp ../data/showdown-config.js ./config/config.js
node pokemon-showdown start --no-security
