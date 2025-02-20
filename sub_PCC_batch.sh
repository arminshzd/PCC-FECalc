#! /bin/zsh

for input in "$@";  do
	echo "Submitting" "$input"
	./run_PCC.sh $input FEN
	./run_PCC.sh $input DEC
done

echo "done."
