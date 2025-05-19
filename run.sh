# rm -f runResults/* 
mkdir -p runResults

# Generate output files
for _ in $(seq 1 5); do
  filename="./runResults/$RANDOM.txt"
  python3 hepatitis-sbgp.py > "$filename"
  echo "Saved output to $filename"
done

# Print header
echo "\n"
echo -e "File\tRandom Seed\tDuration\tBest Fitness"

# Extract last three lines and format as a table
ls ./runResults/ | while read file; do
  seed=$(tail -n 3 "./runResults/$file" | grep "Random Seed" | awk -F '=' '{print $2}' | tr -d ' ')
  duration=$(tail -n 3 "./runResults/$file" | grep "duration" | awk -F '=' '{print $2}' | tr -d ' ')
  fitness=$(tail -n 3 "./runResults/$file" | grep "bestIndivFitness" | awk -F '=' '{print $2}' | tr -d ' ')
  echo -e "$file\t$seed\t$duration\t$fitness"
done
