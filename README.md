# COS710 Assignment 3

This project is a Structure-Based GP algorithm for predicting hepatitis cases using the hepatitis dataset.

## Files Structure

- `hepatitis`: The main executable file with SBGP program
- `hepatitis-sbgp.py`: The main python file with SBGP program code
- `primitives.py`: A python file with class definitions for the functions and terminals used in the program
- `structures.py`: A python file with class definitions for the data structures used in the program
- `hepatitis.tsv`: A `.tsv` file containing the hepatitis dataset


## Running Program

To run the program, run the below command the terminal in the same diractory as the `hepatitis` and `hepatitis.tsv` files:
```bash
$ hepatitis --seed 1
```

> Note: The `--seed` flag is optional. If not provided a random seed between 1 and 100 will be used.

## Arguments

| Name | Defalut Value | Description |
| --- | --- | --- |
| `--seed` | `random.randint(0, 100)` | The psuedo random number generator seed. |

## Author

Seale Rapolai
