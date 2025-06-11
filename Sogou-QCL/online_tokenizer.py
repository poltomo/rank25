from tokenizers import Tokenizer, models, pre_tokenizers, decoders, normalizers
from collections import Counter
import os
import jieba

class OnlineWordLevelTokenizer:
    """
    A tokenizer that supports online training, incremental vocabulary building
    with preservation of old tokens, and optional pruning.
    It uses Hugging Face Tokenizer with a WordLevel model under the hood for performance.
    """

    def __init__(self, initial_vocab_map=None, unk_token="[UNK]", special_tokens=None,
                 length_preference_factor: float = 0.0): # New parameter
        """
        Initializes the tokenizer.

        Args:
            initial_vocab_map (dict, optional): A map of token strings to IDs to initialize the vocabulary.
                                              Defaults to a vocabulary containing only the unk_token.
            unk_token (str, optional): The unknown token string. Defaults to "[UNK]".
            special_tokens (list, optional): A list of special token strings (e.g., "[PAD]", "[CLS]").
                                           UNK token will be added automatically if not present.
            length_preference_factor (float, optional): Controls preference for longer tokens during training.
                                                        0.0: Original behavior (frequency, then alphabetical).
                                                        >0.0: Prefer longer tokens after frequency (e.g., 1.0 for strong preference).
                                                        Defaults to 0.0.
        """
        self.unk_token = unk_token
        self.special_tokens_set = set(special_tokens or [])
        self.special_tokens_set.add(self.unk_token)
        self.length_preference_factor = length_preference_factor # Store the factor

        if initial_vocab_map is None:
            current_id = 0
            vocab = {}
            for s_token in sorted(list(self.special_tokens_set)): # Consistent ordering for IDs
                vocab[s_token] = current_id
                current_id +=1
        else:
            vocab = initial_vocab_map.copy()
            if self.unk_token not in vocab:
                max_id = -1
                if vocab:
                    max_id = max(vocab.values()) if vocab else -1
                vocab[self.unk_token] = max_id + 1
            max_id_in_current_vocab = max(vocab.values()) if vocab else -1
            current_next_id = max_id_in_current_vocab + 1
            for s_token in sorted(list(self.special_tokens_set)):
                if s_token not in vocab:
                    vocab[s_token] = current_next_id
                    current_next_id += 1

        self.tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=self.unk_token))
        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()
        ])
        self.tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        self.tokenizer.decoder = decoders.WordPiece(prefix="##")
        self.tokenizer.add_special_tokens(list(self.special_tokens_set))


    def train_on_chunk(self, text_chunk, min_frequency=2, top_k_new_tokens=500,
                       candidate_type="words", length_preference_factor: float = None): # Override option
        """
        Trains the tokenizer on a new chunk of text, adding new tokens.
        Old tokens are preserved. Token length preference is tunable.

        Args:
            text_chunk (str): The text data to train on.
            min_frequency (int, optional): Minimum frequency for a new token to be added. Defaults to 2.
            top_k_new_tokens (int, optional): Maximum number of new tokens to add from this chunk. Defaults to 500.
            candidate_type (str, optional): Type of candidates to extract: "words" or "ngrams". Defaults to "words".
            length_preference_factor (float, optional): Overrides the instance's default length_preference_factor for this call.
                                                        0.0: Original behavior (frequency, then alphabetical).
                                                        >0.0: Prefer longer tokens after frequency.
                                                        If None, uses the instance's default. Defaults to None.
        """
        if not text_chunk.strip():
            print("Skipping training on empty or whitespace-only chunk.")
            return

        current_length_preference_factor = self.length_preference_factor
        if length_preference_factor is not None:
            current_length_preference_factor = length_preference_factor

        normalized_text = self.tokenizer.normalizer.normalize_str(text_chunk)
        pre_tokenized_segments = [item[0] for item in self.tokenizer.pre_tokenizer.pre_tokenize_str(normalized_text)]
        current_vocab = self.tokenizer.get_vocab()
        candidate_counts = Counter()

        if candidate_type == "words":
            for segment in pre_tokenized_segments:
                if segment not in current_vocab and segment not in self.special_tokens_set:
                    candidate_counts[segment] += 1
        elif candidate_type == "ngrams":
            ngram_range = (2, 5)
            for segment in pre_tokenized_segments:
                if len(segment) < ngram_range[0]: continue
                for n in range(ngram_range[0], min(ngram_range[1] + 1, len(segment) + 1)):
                    for i in range(len(segment) - n + 1):
                        ngram = segment[i:i+n]
                        if ngram not in current_vocab and ngram not in self.special_tokens_set:
                            candidate_counts[ngram] += 1
        else:
            raise ValueError("Invalid candidate_type. Choose 'words' or 'ngrams'.")

        new_token_candidates = []
        # Sort by:
        # 1. Frequency (descending)
        # 2. Scaled token length (descending if factor > 0). If factor is 0, this term becomes 0 for all.
        # 3. Token string (alphabetically, ascending for tie-breaking)
        sorted_candidates = sorted(
            candidate_counts.items(),
            key=lambda x: (-x[1], -current_length_preference_factor * len(x[0]), x[0])
        )

        for token_str, count in sorted_candidates:
            if len(new_token_candidates) >= top_k_new_tokens:
                break
            if count >= min_frequency:
                new_token_candidates.append(token_str)

        if new_token_candidates:
            final_new_candidates = [tok for tok in new_token_candidates if tok not in self.special_tokens_set]
            added_count = self.tokenizer.add_tokens(final_new_candidates)
            print(f"Added {added_count} new tokens. Current vocabulary size: {self.tokenizer.get_vocab_size()}")
        else:
            print("No new tokens met the criteria to be added from this chunk.")

    def tokenize(self, text_sequence):
        if not isinstance(text_sequence, str):
            raise ValueError("Input to tokenize must be a string.")
        output = self.tokenizer.encode(text_sequence)
        return output.ids

    def tokenize_batch(self, text_sequences):
        if not isinstance(text_sequences, list) or not all(isinstance(s, str) for s in text_sequences):
            raise ValueError("Input to tokenize_batch must be a list of strings.")
        output = self.tokenizer.encode_batch(text_sequences)
        return [enc.ids for enc in output]

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

    def get_token_string(self, token_id):
        return self.tokenizer.id_to_token(token_id)

    def get_token_id(self, token_string):
        return self.tokenizer.token_to_id(token_string)

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def prune_tokens(self, tokens_to_prune_strings):
        print(f"Attempting to prune {len(tokens_to_prune_strings)} tokens.")
        current_vocab = self.tokenizer.get_vocab()
        new_vocab = {}
        pruned_count = 0
        for token, token_id in current_vocab.items():
            if token in self.special_tokens_set or token not in tokens_to_prune_strings:
                new_vocab[token] = token_id
            else:
                pruned_count +=1
        if pruned_count == 0 and len(tokens_to_prune_strings) > 0 :
            print("No specified tokens found in vocabulary (or they are special tokens). Nothing pruned.")
            return
        self.tokenizer = Tokenizer(models.WordLevel(vocab=new_vocab, unk_token=self.unk_token))
        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()
        ])
        self.tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        self.tokenizer.decoder = decoders.WordPiece(prefix="##")
        self.tokenizer.add_special_tokens(list(self.special_tokens_set))
        print(f"Pruned {pruned_count} tokens. New vocabulary size: {self.tokenizer.get_vocab_size()}")

    def save(self, directory, name="online_tokenizer"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, f"{name}.json")
        self.tokenizer.save(file_path)
        print(f"Tokenizer saved to {file_path}")

    @classmethod
    def load(cls, file_path, length_preference_factor: float = 0.0): # Added factor for consistency
        """
        Loads a tokenizer from a file.

        Args:
            file_path (str): The path to the tokenizer JSON file.
            length_preference_factor (float, optional): Sets the default length preference
                                                        for the loaded tokenizer instance.
                                                        Defaults to 0.0.

        Returns:
            OnlineWordLevelTokenizer: An instance of this class with the loaded configuration.
        """
        loaded_hf_tokenizer = Tokenizer.from_file(file_path)
        unk_token_from_loaded = loaded_hf_tokenizer.model.unk_token
        vocab_from_loaded = loaded_hf_tokenizer.get_vocab()

        special_tokens_from_loaded_set = set()
        if hasattr(loaded_hf_tokenizer, 'get_added_tokens_decoder'):
            for token_obj in loaded_hf_tokenizer.get_added_tokens_decoder().values():
                if token_obj.special:
                    special_tokens_from_loaded_set.add(str(token_obj.content)) # Use .content

        if unk_token_from_loaded: # Ensure UNK is always part of it for the purpose of init
            special_tokens_from_loaded_set.add(unk_token_from_loaded)

        instance = cls(initial_vocab_map=vocab_from_loaded,
                       unk_token=unk_token_from_loaded,
                       special_tokens=list(special_tokens_from_loaded_set), # Pass the reconstructed list
                       length_preference_factor=length_preference_factor) # Apply factor

        # Ensure the instance's tokenizer has the exact components from the loaded one
        instance.tokenizer.normalizer = loaded_hf_tokenizer.normalizer
        instance.tokenizer.pre_tokenizer = loaded_hf_tokenizer.pre_tokenizer
        instance.tokenizer.decoder = loaded_hf_tokenizer.decoder

        # The special_tokens_set in the instance is built by __init__ based on what's passed.
        # We should ensure it accurately reflects all special tokens in the loaded model.
        final_special_set = set()
        if hasattr(loaded_hf_tokenizer, 'get_added_tokens_decoder'):
            for token_id, added_token_obj in loaded_hf_tokenizer.get_added_tokens_decoder().items():
                if added_token_obj.special:
                    final_special_set.add(str(added_token_obj.content))
        if instance.unk_token:
             final_special_set.add(instance.unk_token) # Ensure UNK from model is there

        instance.special_tokens_set = final_special_set
        # Re-apply add_special_tokens to the internal tokenizer just to be certain it's aware
        # of all tokens our wrapper considers special.
        instance.tokenizer.add_special_tokens(list(instance.special_tokens_set))


        print(f"Tokenizer loaded from {file_path}. Vocab size: {instance.get_vocab_size()}")
        print(f"Instance special_tokens_set after load: {instance.special_tokens_set}")
        print(f"Instance length_preference_factor: {instance.length_preference_factor}")
        return instance




class OnlineJiebaWordLevelTokenizer:
    """
    A robust online tokenizer for Chinese text. It uses Jieba to perform word
    segmentation as a preprocessing step before feeding space-delimited words
    to the underlying WordLevel tokenizer.

    This design is simpler and more reliable than injecting a custom pre-tokenizer.
    """

    def __init__(self, initial_vocab_map=None, unk_token="[UNK]", special_tokens=None,
                 length_preference_factor: float = 0.0):
        """
        Initializes the tokenizer.

        Args:
            initial_vocab_map (dict, optional): A map of token strings to IDs.
            unk_token (str, optional): The unknown token string. Defaults to "[UNK]".
            special_tokens (list, optional): A list of special token strings.
            length_preference_factor (float, optional): Controls preference for longer tokens.
        """
        self.unk_token = unk_token
        self.special_tokens_set = set(special_tokens or [])
        self.special_tokens_set.add(self.unk_token)
        self.length_preference_factor = length_preference_factor

        # --- Vocabulary Setup ---
        if initial_vocab_map is None:
            vocab = {s_token: i for i, s_token in enumerate(sorted(list(self.special_tokens_set)))}
        else:
            vocab = initial_vocab_map.copy()
            next_id = (max(vocab.values()) + 1) if vocab else 0
            for s_token in sorted(list(self.special_tokens_set)):
                if s_token not in vocab:
                    vocab[s_token] = next_id
                    next_id += 1

        self.tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=self.unk_token))

        # --- Configure the Internal Tokenizer for Space-Delimited Input ---
        self.tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.Lowercase()
        ])
        
        # The pre-tokenizer now simply splits by whitespace.
        # Our Python methods will provide it with correctly formatted strings.
        self.tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        # *** THE FIX IS HERE: Use decoders.WordPiece() as the correct replacement. ***
        self.tokenizer.decoder = decoders.WordPiece()

        self.tokenizer.add_special_tokens(list(self.special_tokens_set))

    def _preprocess_text(self, text_chunk: str) -> str:
        """Uses jieba to segment text and joins with spaces."""
        return " ".join(jieba.lcut(text_chunk, cut_all=False))

    def train_on_chunk(self, text_chunk: str, min_frequency=2, top_k_new_tokens=500,
                       length_preference_factor: float = None):
        """
        Trains the tokenizer on a new chunk of Chinese text.
        """
        if not text_chunk.strip():
            print("Skipping training on empty or whitespace-only chunk.")
            return

        processed_chunk = self._preprocess_text(text_chunk)
        
        pre_tokenized_segments = [item[0] for item in self.tokenizer.pre_tokenizer.pre_tokenize_str(processed_chunk)]
        
        current_vocab = self.get_vocab()
        candidate_counts = Counter()
        for segment in pre_tokenized_segments:
            if segment not in current_vocab and segment not in self.special_tokens_set:
                candidate_counts[segment] += 1
        
        current_lpf = self.length_preference_factor if length_preference_factor is None else length_preference_factor
        sorted_candidates = sorted(
            candidate_counts.items(),
            key=lambda x: (-x[1], -current_lpf * len(x[0]), x[0])
        )

        new_token_candidates = [
            token_str for token_str, count in sorted_candidates
            if count >= min_frequency
        ][:top_k_new_tokens]

        if new_token_candidates:
            added_count = self.tokenizer.add_tokens(new_token_candidates)
            print(f"Added {added_count} new tokens. Current vocabulary size: {self.get_vocab_size()}")
        else:
            print("No new tokens met the criteria to be added from this chunk.")

    def tokenize(self, text_sequence: str) -> list[int]:
        """Tokenizes a single Chinese text sequence into token IDs."""
        processed_sequence = self._preprocess_text(text_sequence)
        output = self.tokenizer.encode(processed_sequence)
        return output.ids
    
    def tokenize_batch(self, text_sequences: list[str]) -> list[list[int]]:
        """Tokenizes a batch of Chinese text sequences."""
        processed_sequences = [self._preprocess_text(s) for s in text_sequences]
        output = self.tokenizer.encode_batch(processed_sequences)
        return [enc.ids for enc in output]

    def decode(self, token_ids: list[int]) -> str:
        """Decodes a list of token IDs back into a space-delimited string."""
        return self.tokenizer.decode(token_ids)
    
    def get_vocab(self): return self.tokenizer.get_vocab()
    def get_vocab_size(self): return self.tokenizer.get_vocab_size()
    def save(self, directory, name="chinese_tokenizer"):
        if not os.path.exists(directory): os.makedirs(directory)
        file_path = os.path.join(directory, f"{name}.json")
        self.tokenizer.save(file_path)
        print(f"Tokenizer saved to {file_path}")

    @classmethod
    def load(cls, file_path, length_preference_factor: float = 0.0):
        loaded_hf_tokenizer = Tokenizer.from_file(file_path)
        unk_token = loaded_hf_tokenizer.model.unk_token
        vocab = loaded_hf_tokenizer.get_vocab()
        special_tokens = {
            str(tok.content) for tok in loaded_hf_tokenizer.get_added_tokens_decoder().values() if tok.special
        }
        instance = cls(
            initial_vocab_map=vocab,
            unk_token=unk_token,
            special_tokens=list(special_tokens),
            length_preference_factor=length_preference_factor
        )
        print(f"Tokenizer loaded from {file_path}. Vocab size: {instance.get_vocab_size()}")
        return instance