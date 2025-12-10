import json 
def read_json(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            ltext , lsize = [], []
            for i, x in enumerate(file, 1):
                #stripping whitespaces from line
                line = x.strip()
                
                
                #line is empty
                if not line:
                    continue

                try:
                    s = json.loads(line)

                    text = s.get("Text", "N/A")
                    size = s.get("Size", "N/A")

                    duo = (text, size)
                    data.append(duo)

                    #data = [text,size]
                    #print(f"Text Line {i}: {text} , size: {size}")

                except Exception as e:
                    print("Error decoding line")
        
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding file: {file_path}")

def main():
    parsed= read_json("/mnt/c/Users/jimso/MyCode/Termination Project/data/seed_slots.jsonl")
    for list in parsed:
            print(list)
if __name__ == "__main__":
    main()