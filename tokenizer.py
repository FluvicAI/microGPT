import re

class Tokenizer:
    def __init__(self):
        self.merges = {}
        # for now this simple pre-tokenizer regex
        self.regex = r'(\b\w+\W*|\W+\b|\S+|\s+)'
        # should be added after generating merge tree, initialize BEFORE merge tree generation
        self.special_tokens = {}
        # should be added before generating merge tree, initialize BEFORE merge tree generation
        self.manual_tokens = {}
        
    def generate_merge_tree(self, num_merges, ids, s, log=False):
        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = i + s 
            ids = self.merge(ids, pair, idx)

            # logging in tree
            self.merges[pair] = idx

            if log == True:
                print('Merging: [' + (chr(pair[0]) if pair[0] < 256 else str(pair[0]) + ' ') + (chr(pair[1]) if pair[1] < 256 else str(pair[1])) + '] into ' + str(idx))

        return ids
    
    def process_input(self, input_path, vocab_size, output_path, log=False, eval_int=1):
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
        
        text_split = re.findall(self.regex, text, re.UNICODE)

        # add manual tokens
        self.merges.update(self.manual_tokens)

        s = 256 + len(self.manual_tokens)
        
        merges_total = vocab_size - s - len(self.special_tokens)
        ids = [list(i.encode('utf-8')) for i in text_split]

        ## HERE
        for i in range(merges_total):
            ids = self.generate_merge_tree(1, ids, i + s)
            if i%eval_int == 0:
                print(f'Step {i} of {merges_total}')
                save_dict = {}
                save_dict['merges'] = self.merges
                save_dict['regex'] = self.regex

                with open(output_path, 'w') as file:
                    file.write(str(save_dict))

        # add special tokens
        self.merges.update(self.special_tokens)

        save_dict = {}
        save_dict['merges'] = self.merges
        save_dict['regex'] = self.regex

        with open(output_path, 'w') as file:
            file.write(str(save_dict))

        print('Compression ratio: ' + str(vocab_size/256))
        print('Dictionary saved to: ' + output_path)


        return self.merges
    
    def add_dict(self,d1,d2):
        for key,val in d2.items():
            if key in d1:
                d1[key] += d2[key]
            else:
                d1[key] = val
        return d1

    def load_dict(self, dict):
        self.regex = dict['regex']
        self.merges = dict['merges']
        self.vocab_size = len(self.merges) + 256

        return

    def get_stats(self, ids_arr):
        counts = {}
        for ids in ids_arr:
            d = {}
            for pair in zip(ids, ids[1:]):
                d[pair] = d.get(pair,0) + 1
            self.add_dict(counts, d)
            
        return counts
    
    def merge(self, ids_arr, pair, idx):
        newids_arr = []
        for ids in ids_arr:
            newids = []
            i=0
            while i<len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                    newids.append(idx)
                    i += 2
                else:
                    newids.append(ids[i])
                    i += 1
            newids_arr.append(newids)

        return newids_arr
    
    def encode_split(self, text):
        tokens = list(text.encode('utf-8'))
        
        if len(tokens) < 2:
            return tokens
        while True:
            stats = self.get_stats([tokens])
            if(len(stats)) == 0:
                break
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            
            tokens = self.merge([tokens], pair, self.merges[pair])[0]
        return tokens

    def encode(self, text):
        text_split = re.findall(self.regex, text, re.UNICODE)
        tokens = []
        [tokens.extend(self.encode_split(i)) for i in text_split]

        return tokens
    
    def decode(self, ids):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0,p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        tokens = b"".join(vocab[idx] for idx in ids)
        text = tokens.decode('utf-8', errors='replace')
        return text
    
    def get_state(self):
        save_dict = {}
        save_dict['merges'] = self.merges
        save_dict['regex'] = self.regex
        return save_dict
    
    def test(self, text, dark_mode=True):
        # displays tokenization, 
        # when this function displays '?',
        # it means a character is split up into tokens, 
        # but the real decode function should handle it correctly
        ids = self.encode(text)
        
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0,p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
            
        newtext = ''
        
        for i, idx in enumerate(ids):
            color = '\u001b['
            if dark_mode:
                color = '\u001b[37;'
            if i%5 == 0:
                color = color + '41m'
            elif i%5 == 1:
                color = color + '44m'
            elif i%5 == 2:
                color = color + '43m'
            elif i%5 == 3:
                color = color + '46m'
            elif i%5 == 4:
                color = color + '42m'      
            newtext = newtext + color + vocab[idx].decode('utf-8', errors='replace') + '\u001b[0m'
        
        return newtext