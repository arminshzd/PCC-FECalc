#! /bin/zsh

for PCC in "$@";
do
	echo "$PCC"_FEN: $( tail -n 1 "$PCC"_FEN.out )
        echo "$PCC"_DEC: $( tail -n 1 "$PCC"_DEC.out )

done
