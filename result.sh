LOG_PATH=/home/ultraman/PeilinWu/Repos/pytorch-models/saved/log
GREP_RESULT=grep.log
cd $LOG_PATH

for d in */; do
    CUR_PATH="$LOG_PATH/$d"
    cd $CUR_PATH
    TIME_PATH=*/;
    CUR_PATH=`echo $CUR_PATH${TIME_PATH[0]}`
    LOG=$CUR_PATH
    LOG+='info.log'
    echo $LOG
    GREP_FILE=$CUR_PATH$GREP_RESULT
    rm $GREP_FILE
    touch $GREP_FILE
    printf "val_loss:\n" >> $GREP_FILE
    cat $LOG | grep "val_loss       :" |  grep -Eo '[0-9]+[.][0-9]+' >> $GREP_FILE
    printf "\nval_weighted_loss:\n" >> $GREP_FILE
    cat $LOG | grep "val_weighted_loss:" |  grep -Eo '[0-9]+[.][0-9]+' >> $GREP_FILE
    printf "\npareto:\n" >> $GREP_FILE
    PARETO_ARRAY=`cat $LOG | grep "pareto: " | tail -n 1 | grep -Eo "\[.*\]$"`
    echo $PARETO_ARRAY | sed 's/\[//g' | sed 's/\]\]//g' | sed 's/\], /\n/g' | sed 's/, /\t/g' >> $GREP_FILE
done