import csv
import io
import pandas as pd

cleaned_lines = []
with open('random_20000.csv', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        # Bahar ki quotes hata do
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        cleaned_lines.append(line)

content = '\n'.join(cleaned_lines)

# csv reader se parse karo
reader = csv.reader(io.StringIO(content))
rows = list(reader)

headers = rows[0]
print("Headers:", headers)
print("Total columns:", len(headers))

# Sirf utni columns wali rows lo
num_cols = len(headers)
clean_rows = [r for r in rows[1:] if len(r) == num_cols]
skipped = len(rows) - 1 - len(clean_rows)
print(f"Skipped rows: {skipped}")

df = pd.DataFrame(clean_rows, columns=headers)
df.columns = df.columns.str.strip()

print("\nColumns:", df.columns.tolist())
print("\nGenere value counts:")
print(df['genere'].value_counts().head(20))
# Clean CSV save karo
df.to_csv('clean_movies.csv', index=False)
print(f"\nClean CSV save ho gayi! Total rows: {len(df)}")