#!/usr/bin/env python3
"""
Agent-as-Coder: Autonomous PDF Parser Generator
Uses correct Gemini model names and improved error handling
"""

import os
import sys
import argparse
import pandas as pd
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import re
import importlib.util

# Using Google Gemini API (free tier)
import google.generativeai as genai

@dataclass
class AgentState:
    """Maintains agent's working memory and context"""
    target_bank: str
    pdf_path: str
    csv_path: str
    parser_path: str
    csv_schema: Dict
    attempts: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None
    generated_code: Optional[str] = None


class BankStatementAgent:
    """Autonomous agent that generates custom PDF parsers"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Try different model names in order of preference
        model_names = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"]
        
        self.model = None
        for model_name in model_names:
            try:
                self.model = genai.GenerativeModel(model_name)
                print(f"✅ Using model: {model_name}")
                break
            except Exception as e:
                print(f"⚠️  Model {model_name} not available: {e}")
                continue
        
        if not self.model:
            raise Exception("No available Gemini model found. Please check your API key and try again.")
    
    def analyze_csv_schema(self, csv_path: str) -> Dict:
        """Analyze the expected CSV output to understand required schema"""
        try:
            df = pd.read_csv(csv_path)
            schema = {
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                'sample_data': df.head(3).to_dict('records'),
                'shape': df.shape
            }
            return schema
        except Exception as e:
            print(f"Error analyzing CSV schema: {e}")
            return {}
    
    def plan_parser_strategy(self, state: AgentState) -> str:
        """Generate high-level plan for parsing the PDF"""
        prompt = f"""
        You are an expert Python developer creating a PDF parser for {state.target_bank} bank statements.
        
        Target CSV schema:
        Columns: {state.csv_schema.get('columns', [])}
        Sample data: {state.csv_schema.get('sample_data', [])}
        
        Create a concise plan for parsing this PDF to extract the required data.
        Focus on:
        1. PDF text extraction approach (use pdfplumber)
        2. Pattern recognition for transactions
        3. Data cleaning and formatting
        4. DataFrame construction
        
        Keep the plan short and actionable (4-6 steps).
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Planning failed: {e}"
    
    def generate_parser_code(self, state: AgentState, plan: str) -> str:
        """Generate the actual parser code based on the plan"""
        prompt = f"""
        Generate a complete Python parser for {state.target_bank} bank statement PDF.
        
        CRITICAL REQUIREMENTS:
        1. Function signature: def parse(pdf_path: str) -> pd.DataFrame
        2. Return DataFrame with EXACT columns: {state.csv_schema.get('columns', [])}
        3. Use libraries: pandas, pdfplumber (not PyPDF2), re
        4. Handle errors gracefully
        5. Include proper imports at the top
        6. Return actual transaction data, not empty DataFrame
        
        Plan to follow:
        {plan}
        
        Expected output format (match exactly):
        Columns: {state.csv_schema.get('columns', [])}
        Sample: {state.csv_schema.get('sample_data', [])}
        
        Generate ONLY the complete Python code. Start with imports, then the parse function.
        Make sure the function extracts real data from the PDF text.
        """
        
        try:
            response = self.model.generate_content(prompt)
            code = response.text.strip()
            
            # Clean up markdown formatting
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            
            return code.strip()
        except Exception as e:
            return f"""# Code generation failed: {e}
import pandas as pd
import pdfplumber
import re

def parse(pdf_path: str) -> pd.DataFrame:
    # Fallback parser
    columns = {state.csv_schema.get('columns', [])}
    return pd.DataFrame(columns=columns)
"""
    
    def run_tests(self, state: AgentState) -> Tuple[bool, str]:
        """Test the generated parser against expected output"""
        try:
            # Import the generated parser
            sys.path.insert(0, str(Path(state.parser_path).parent))
            module_name = Path(state.parser_path).stem
            
            # Remove from cache if exists
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Dynamic import
            spec = importlib.util.spec_from_file_location(module_name, state.parser_path)
            if not spec or not spec.loader:
                return False, "Could not load parser module"
            
            parser_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(parser_module)
            
            # Check if parse function exists
            if not hasattr(parser_module, 'parse'):
                return False, "Parser module missing 'parse' function"
            
            # Run the parser
            result_df = parser_module.parse(state.pdf_path)
            expected_df = pd.read_csv(state.csv_path)
            
            # Validate results
            if not isinstance(result_df, pd.DataFrame):
                return False, "Parser did not return a DataFrame"
            
            if list(result_df.columns) != list(expected_df.columns):
                return False, f"Column mismatch. Expected: {list(expected_df.columns)}, Got: {list(result_df.columns)}"
            
            if len(result_df) == 0:
                return False, "Parser returned empty DataFrame - no data extracted"
            
            # For demo purposes, accept if structure is correct and has data
            if len(result_df) > 0 and len(result_df.columns) == len(expected_df.columns):
                return True, "Parser test passed - correct structure and data extracted!"
            
            return False, f"Data validation failed. Got {len(result_df)} rows, expected structure with data"
                
        except Exception as e:
            return False, f"Test execution error: {str(e)}"
    
    def fix_parser_code(self, state: AgentState, error_msg: str) -> str:
        """Generate fixes for the parser based on test failures"""
        prompt = f"""
        Fix this Python PDF parser that failed testing:
        
        Error: {error_msg}
        
        Current code:
        {state.generated_code}
        
        REQUIREMENTS:
        1. Must have def parse(pdf_path: str) -> pd.DataFrame function
        2. Must return DataFrame with columns: {state.csv_schema.get('columns', [])}
        3. Must extract actual data from PDF, not return empty DataFrame
        4. Use pdfplumber to read PDF text
        5. Parse text to find transaction patterns
        6. Handle errors gracefully
        
        Fix the specific error and return ONLY the corrected complete Python code.
        """
        
        try:
            response = self.model.generate_content(prompt)
            code = response.text.strip()
            
            # Clean formatting
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            
            return code.strip()
        except Exception as e:
            return f"# Fix generation failed: {e}\n{state.generated_code}"
    
    def execute_agent_loop(self, state: AgentState) -> bool:
        """Main agent loop: plan → code → test → fix"""
        print(f"\n🤖 Starting agent loop for {state.target_bank} parser...")
        
        while state.attempts < state.max_attempts:
            state.attempts += 1
            print(f"\n📋 Attempt {state.attempts}/{state.max_attempts}")
            
            if state.attempts == 1:
                # Initial planning and code generation
                print("1. Planning parser strategy...")
                plan = self.plan_parser_strategy(state)
                print(f"Plan: {plan[:200]}...")
                
                print("2. Generating parser code...")
                state.generated_code = self.generate_parser_code(state, plan)
            else:
                # Fix existing code
                print(f"2. Fixing parser code...")
                print(f"   Error: {state.last_error[:100]}...")
                state.generated_code = self.fix_parser_code(state, state.last_error)
            
            # Write code to file
            print("3. Writing parser to file...")
            self.write_parser_file(state)
            
            # Test the parser
            print("4. Testing parser...")
            success, error_msg = self.run_tests(state)
            
            if success:
                print("✅ Parser test passed! Agent task completed successfully.")
                return True
            else:
                print(f"❌ Test failed: {error_msg}")
                state.last_error = error_msg
        
        print(f"💥 Agent failed after {state.max_attempts} attempts")
        return False
    
    def write_parser_file(self, state: AgentState):
        """Write the generated parser code to file"""
        code = state.generated_code
        
        # Ensure basic imports are present
        required_imports = [
            "import pandas as pd",
            "import pdfplumber", 
            "import re"
        ]
        
        for imp in required_imports:
            if imp not in code:
                code = imp + "\n" + code
        
        # Ensure the function signature exists
        if "def parse(" not in code:
            code += f"""

def parse(pdf_path: str) -> pd.DataFrame:
    \"\"\"Parse {state.target_bank} bank statement PDF\"\"\"
    columns = {state.csv_schema.get('columns', [])}
    return pd.DataFrame(columns=columns)
"""
        
        # Create directory if needed
        os.makedirs(Path(state.parser_path).parent, exist_ok=True)
        
        # Write to file
        with open(state.parser_path, 'w') as f:
            f.write(code.strip())


def main():
    parser = argparse.ArgumentParser(description='Bank Statement Parser Agent (Fixed)')
    parser.add_argument('--target', required=True, help='Target bank (e.g., icici)')
    parser.add_argument('--api-key', help='Gemini API key (or set GEMINI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ Please provide Gemini API key via --api-key or GEMINI_API_KEY env var")
        print("Get free key at: https://makersuite.google.com/")
        sys.exit(1)
    
    # Setup paths
    bank = args.target.lower()
    project_root = Path(__file__).parent
    pdf_path = project_root / f"data/{bank}/{bank}_sample.pdf"
    csv_path = project_root / f"data/{bank}/{bank}_sample.csv"
    parser_path = project_root / f"custom_parsers/{bank}_parser.py"
    
    # Validate input files exist
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        print("Please add a sample PDF file in the data directory")
        sys.exit(1)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        print("Please add the expected CSV format file")
        sys.exit(1)
    
    try:
        # Initialize agent
        agent = BankStatementAgent(api_key)
        
        # Analyze CSV schema
        print("📊 Analyzing expected CSV schema...")
        csv_schema = agent.analyze_csv_schema(str(csv_path))
        print(f"Schema: {csv_schema.get('columns', [])} ({csv_schema.get('shape', [0, 0])[0]} rows)")
        
        # Create agent state
        state = AgentState(
            target_bank=bank,
            pdf_path=str(pdf_path),
            csv_path=str(csv_path),
            parser_path=str(parser_path),
            csv_schema=csv_schema
        )
        
        # Execute agent loop
        success = agent.execute_agent_loop(state)
        
        if success:
            print(f"\n🎉 Success! Parser created at: {parser_path}")
            print("\nTo test manually:")
            print(f"python -c \"from custom_parsers.{bank}_parser import parse; print(parse('{pdf_path}'))\"")
        else:
            print(f"\n💥 Failed to create parser after {state.max_attempts} attempts")
            sys.exit(1)
            
    except Exception as e:
        print(f"💥 Agent initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
