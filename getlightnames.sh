#!/bin/bash

rm -f lightlist0.tmp lightlist1.tmp lightlist.tmp
cat *light* | tr '\n\r' '  ' | sed 's/  */\n/g' | grep -e '[a-z]:[a-z]' | sort -u > lightlist0.tmp
cat shaders/block.properties | sed 's/#.*$//' | sed 's/^.*[0-9]=//' | tr -d '\r' | tr ' ' '\n' | grep -e '[a-z]' | sort -u >> lightlist1.tmp
KNOWNLIGHTS=$(cat lightlist1.tmp)
for L in $KNOWNLIGHTS; do
    cat lightlist0.tmp | sed "s/^$L\$//" > lightlist1.tmp
    mv lightlist1.tmp lightlist0.tmp
done
cat lightlist0.tmp | tr '\n' ' ' | sed 's/  */ /g' > lightlist.tmp
echo '' >> lightlist.tmp
cat shaders/block.properties lightlist.tmp | sed 's/#.*$//' | sed 's/^.*[0-9]=//' | tr -d '\r' | tr ' ' '\n' | grep -e '[a-z]' | sort | uniq -d
