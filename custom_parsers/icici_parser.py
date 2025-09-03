import pandas as pd
import pdfplumber
import re
from datetime import datetime

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parses an ICICI bank statement PDF and returns a Pandas DataFrame.

    Args:
        pdf_path: Path to the ICICI bank statement PDF.

    Returns:
        A Pandas DataFrame with columns ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'].
        Returns an empty DataFrame if parsing fails.
    """

    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

        # Regular expression to identify transaction lines (adjust as needed based on PDF format variations)

        lines = re.findall(r"(\d{2}-\d{2}-\d{4})\s+([\w\s\.\,\-\&]+)\s+([\d,.]+)?\s+([\d,.]+)?\s+([\d,.]+)", text)


        data = []
        for line in lines:
            date_str = line[0]
            description = line[1].strip()
            debit = line[2].strip() if line[2].strip() else None
            credit = line[3].strip() if line[3].strip() else None
            balance = line[4].strip()


            try:
                date = datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y-%m-%d")
            except ValueError:
                date = None


            try:
                debit = float(debit.replace(",", "")) if debit else None
            except (ValueError, AttributeError):
                debit = None

            try:
                credit = float(credit.replace(",", "")) if credit else None
            except (ValueError, AttributeError):
                credit = None

            try:
                balance = float(balance.replace(",", ""))
            except (ValueError, AttributeError):
                balance = None

            data.append({
                'Date': date,
                'Description': description,
                'Debit Amt': debit,
                'Credit Amt': credit,
                'Balance': balance
            })

        df = pd.DataFrame(data)
        return df

    except FileNotFoundError:
        print(f"Error: File not found at {pdf_path}")
        return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])