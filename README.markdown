# HiFBA: High-Order Inexact Forward-Backward Algorithm

HiFBA is an algorithm for sparse signal recovery with Laplace noise, designed for solving inverse problems with a non-smooth regularizer. This implementation runs the algorithm for a fixed `q=1.5` with predefined parameters.

## Author

- **Name**: Alireza Kabgani
- **Email**: a.kabgani@gmail.com
- **Website**: https://sites.google.com/view/akabgani

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/akabgani/HiFBA.git
   cd hifba
   ```

2. Install the required dependencies:

   ```bash
   pip install numpy matplotlib
   ```

## Usage

To run the HiFBA algorithm with `q=1.5` and default parameters:

```bash
python src/main.py
```

This will generate a test problem, run the HiFBA algorithm, and display results including SNR, relative error, and convergence plots.

## Project Structure

- `src/`
  - `__init__.py`: Package initialization and metadata.
  - `problem.py`: Generates test problems for sparse signal recovery.
  - `algorithm.py`: Implements the HiFBA algorithm.
  - `utils.py`: Utility functions for SNR computation and visualization.
  - `main.py`: Main script to run the algorithm for `q=1.5`.
- `LICENSE`: MIT License for the project.
- `README.md`: This file.

## Requirements

- Python 3.6+
- NumPy
- Matplotlib

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue on GitHub.

## Contact

For questions or feedback, contact Alireza Kabgani at a.kabgani@gmail.com.