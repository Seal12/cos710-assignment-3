if [ -d "./runResults/" ]; then
  mkdir runResults
fi

for _ in $(seq 1 10); do
  filename="./runResults/$RANDOM.txt"
  python3 hepatitis-sbgp.py > "$filename"
  echo "Saved output to $filename"
done

ls ./runResults/ | while read file; do   first_line=$(head -n 1 "./runResults/$file");   echo "$first_line $file"
