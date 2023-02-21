#!/bin/bash
if [ $1 ];then
    SETTINGSFILE=$1
else
    if [ $(pwd | grep rethinking) ];then
    SETTINGSFILE=$(find . -name common.glsl)
    else
    SETTINGSFILE=$(find . -name common.glsl | grep rethinking)
    fi
fi
if [ $2 ];then
    PROPFILE=$2
else
    if [ $(pwd | grep rethinking) ];then
        PROPFILE=$(find . -name shaders.properties)
    else
        PROPFILE=$(find . -name shaders.properties | grep rethinking)
    fi
fi
cat $SETTINGSFILE | grep -e '#define [^\.]*$' | sed 's/^.*#define //' | sort -u > settings_with_values.tmp
cat $PROPFILE | grep -e 'screen\.' | sed 's/^.*screen\..*=//' | sed 's/\[[A-Z_]*\]//g' | sed 's/<empty>//g' | sed 's/  */ /g' | sed 's/^ //' | tr ' ' '\n' | sort -u > menu_settings.tmp
SETTINGS_WITHOUT_VALUES=$(cat settings_with_values.tmp | sed 's/ .*$//')
rm -f validsettings0.tmp
for SETTING in $(cat menu_settings.tmp);do
    VALID=
    for SETTING2 in $SETTINGS_WITHOUT_VALUES;do
        if [ $SETTING == $SETTING2 ];then
        VALID=1
        fi
    done
    if [ $VALID ];then
        echo $SETTING >> validsettings0.tmp
    fi
done
CANCONTINUE=not_yet
while [ $CANCONTINUE == not_yet ];do
    if [ $(ls validsettings.txt 2>/dev/null) ];then
        CANCONTINUE=yes
    fi
    sleep 10
done
VALIDSETTINGS=$(cat validsettings.txt)
cp rethinking-voxels.txt rethinking_voxels_old.txt
for SETTING1 in $VALIDSETTINGS;do
    VALUES1=$(cat settings_with_values.tmp | grep "^$SETTING1" | sed 's/^.*\[//' | sed 's/\].*$//')
    if [ "$SETTING1" == "$VALUES1" ];then
        VALUES1="true false"
    fi
    echo "still here!"
    for SETTING2 in $VALIDSETTINGS;do
        if [ "$SETTING2" != "$SETTING1" ];then
            VALUES2=$(cat settings_with_values.tmp | grep "^$SETTING2" | sed 's/^.*\[//' | sed 's/\].*$//')
            if [ "$SETTING2" == "$VALUES2" ];then
                VALUES2="true false"
            fi
            for SETTING3 in $VALIDSETTINGS;do
                if [ "$SETTING3" != "$SETTING1" ];then
                    if [ "$SETTING3" != "$SETTING2" ];then
                        VALUES3=$(cat settings_with_values.tmp | grep "^$SETTING3" | sed 's/^.*\[//' | sed 's/\].*$//')
                        if [ "$SETTING3" == "$VALUES3" ];then
                            VALUES3="true false"
                        fi
                        for VALUE1 in $VALUES1;do
                            for VALUE2 in $VALUES2;do
                                for VALUE3 in $VALUES3;do
                                    echo "$SETTING1=$VALUE1
$SETTING2=$VALUE2
$SETTING3=$VALUE3
                                    " > rethinking-voxels.txt
                                    echo "$SETTING1=$VALUE1
$SETTING2=$VALUE2
$SETTING3=$VALUE3
                                    "
                                    rm -f ~/.minecraft/logs/latest.log
                                    touch ~/.minecraft/logs/latest.log
                                    xdotool keydown r
                                    xdotool keyup r
                                    CANCONTINUE=not_yet
                                    sleep 4
                                    if [ $(cat ~/.minecraft/logs/latest.log | grep "Shader compilation log") ];then
                                        cat ~/.minecraft/logs/latest.log
                                        cp rethinking_voxels_old.txt rethinking-voxels.txt
                                        exit
                                    fi
                                done
                            done
                        done
                    fi
                fi
            done
        fi
    done
done
mv rethinking_voxels_old.txt rethinking-voxels.txt
#rm *.tmp