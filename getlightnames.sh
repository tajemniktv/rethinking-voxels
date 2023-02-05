#!/bin/bash

rm -f lightlist0.tmp lightlist.tmp
cat *light* | tr '\n\r' '  ' | sed 's/  */\n/g' | grep -e '[a-z]:[a-z]' >> lightlist0.tmp
cat shaders/block.properties | grep -e 'block.1234=' | sed 's/^block.1234=//' | tr -d '\r' |tr ' ' '\n' >> lightlist0.tmp
cat lightlist0.tmp | sort -u | tr '\n' ' ' > lightlist.tmp
rm lightlist0.tmp
echo '' >> lightlist.tmp
cat shaders/block.properties | sed 's/#.*$//' | sed 's/^.*[0-9]=//' | tr -d '\r' | tr ' ' '\n'# | grep -e '[a-z]' | sort | uniq -d
