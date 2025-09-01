# Agent-as-Coder: Bank Statement Parser Generator

An autonomous coding agent that generates custom PDF parsers for bank statements using LLM-driven planning and iterative refinement.

## Quick Start (5 Steps)

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd ai-agent-challenge
   pip install -r requirements.txt
   ```

2. **Set API key**:
   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   ```

3. **Run agent for ICICI**:
   ```bash
   python agent.py --target icici
   ```

4. **Verify parser creation**:
   ```bash
   ls custom_parsers/icici_parser.py
   ```

5. **Run tests**:
   ```bash
   pytest test_agent.py -v
   ```

## Agent Architecture

The agent follows a **Plan → Code → Test → Fix** loop:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│   ANALYZE   │───▶│     PLAN     │───▶│  GENERATE   │───▶│    TEST     │
│ CSV Schema  │    │  Strategy    │    │    Code     │    │   Parser    │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
                                              ▲                   │
                                              │         FAIL      ▼
                                         ┌─────────────────────────────┐
                                         │          FIX CODE          │
                                         │    (Max 3 attempts)        │
                                         └─────────────────────────────┘
```

The agent maintains state across iterations, learns from failures, and autonomously refines its approach until the parser passes all tests or reaches the maximum attempt limit.

## Usage Examples

### Generate ICICI parser:
```bash
python agent.py --target icici
```

### Generate parser for new bank (SBI):
```bash
# Place SBI sample files in data/sbi/
python agent.py --target sbi
```

### Test existing parser:
```bash
python -c "from custom_parsers.icici_parser import parse; print(parse('data/icici/icici_sample.pdf'))"
```

## Project Structure

```
├── agent.py                 # Main agent implementation
├── test_agent.py           # Test suite
├── requirements.txt        # Dependencies
├── data/
│   └── icici/
│       ├── icici_sample.pdf   # Sample bank statement
│       └── icici_sample.csv   # Expected output format
└── custom_parsers/
    └── icici_parser.py        # Generated parser (created by agent)
```

## Features

- **Autonomous Operation**: No manual intervention required
- **Self-Debugging**: Agent iteratively fixes its own code
- **Schema-Aware**: Analyzes expected output format automatically  
- **Extensible**: Works with any bank given sample PDF/CSV pair
- **Robust Testing**: Validates parser output against expected results
- **Clean Architecture**: Modular design with clear separation of concerns

## API Requirements

Get free API key from [Google AI Studio](https://makersuite.google.com/) and set as environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
```
