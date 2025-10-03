

## Goal

Find subset-minimal sets of neuron-status literals.

**Formal statement:**

![equation](assets/formal_goal.png)

Where:
- S = set of literals (active/inactive neurons)
- R = region around x₀ (e.g. L∞ ball)
- c = predicted class at x₀
### Method (oracle + greedy)
- `check(S)` uses a verifier (MILP/BaB) with status constraints on \(S\) and region constraints on \(R\).
- Start from full NAP, greedily remove literals while preserving certification.
  

### 
- Minimality of  the explanation depends on the chosen traversal heuristic and is not guaranteed as the minimal cardinality.


## Features

*Neural Activation Extraction
Extract the neural activations produced when the model performs inference on a given input.

*Local Robust Region Identification
Determine the largest surrounding region (an ε-ball around the input) in which the extracted neural activation pattern ensures the robustness of the model’s prediction.

*Constraint Relaxation via Formal Verification
Use a formal verifier to selectively relax certain neuron activation constraints, while still preserving the original robustness property within the region.

*Neuron Constraint Prioritization
The user can configure priority rules for dropping activation constraints, based on a set of implemented heuristics.
Each heuristic affects the minimal neuron set found — i.e., the smallest subset of neurons such that abstracting their activation states within the region changes the model’s prediction.

Timeout Configuration
The user can specify a timeout for the verification process when attempting to drop the activation constraint on a particular neuron.

---

## Project Structure

```
root/
 ├── src/          # Source code
 ├── docs/         # Documentation
 ├── tests/        # Unit tests
 ├── data/         # Data files
 └── README.md     # This file
```

---

## Installation

### Prerequisites

- Language/Framework version
- Dependencies

### Steps

```bash
# Clone the repository
git clone https://github.com/something
cd project

# Install dependencies
pip install -r requirements.txt   # or npm install / gradle build etc.

# Run the script
python main.py
```

---

## Usage

```bash
# Example command to run
python main.py --option value
```

## Screenshots

Here is the homepage of the project:

![Homepage Screenshot](assets/homepage.png)
demos

## Tests

```bash
pytest tests/    # or npm test / gradle test etc.
```

---

## Roadmap

- [ ] Feature A
- [ ] Feature B
- [ ] Feature C

---

## FAQ

- **Q** : Why do we only consider fully connected networks whith RELU activation functions ?  
  **R** : I will answer it later .

- **Q** : Can we really find the smallest set of neurons that define the explanation ?  
  **R** : I will answer it later and add more questions

## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/foo`)
3. Commit your changes (`git commit -m 'Add foo'`)
4. Push to the branch (`git push origin feature/foo`)
5. Create a new Pull Request

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

---

## Author(s)

- **Sarah** – [@username](https://github.com/SarahDribi)
