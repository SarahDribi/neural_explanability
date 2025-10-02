# Project Title

Minimal neural based explanation of relu-based neural networks
with formal verification garantees

---

## Features

- Extraction of neural activations that happened when the model infered on the input
- Finding the maximum biggest region around input (epsilon ball) where the local neural activation
  extracted enforces more robustness of the model's prediction
- Calling a formal verifier to loosen some of the neural activation constraints ,
  while still keeping the intial robustness property around the region
  -The user can set the priority of the neuron constraints dropping , following a
  set of implemented heuristics , each heuristic impacts the found minimal neurons set:
  the shortest set of neurons given this heuristic traversal such that abstracting one's
  state in this input region changes the classifier's prediction
- The timeout of the checking process for dropping the activation constraint on a certain
  neuron can be specified by the user .

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
