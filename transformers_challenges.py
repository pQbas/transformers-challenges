import torch
import math
import torch.nn.functional as F
import torch.nn as nn

class Challenge1:
    @staticmethod
    def test(cls):
        try:
            inputs = Challenge1.inputs()
            instance = cls(inputs['vocab_size'], inputs['d_model'])  # Instantiate the user's class
            output = instance(inputs['example_input'])  # Run forward pass
            
            # Check if output shape matches expected shape
            assert output.shape == (2, 5, inputs['d_model']), f"Expected output shape (2, 5, {inputs['d_model']}), but got {output.shape}"
            print("Challenge 1 passed!")
        except AttributeError as e:
            print(f"Challenge 1 failed: Missing or incorrect attribute or method. {str(e)}")
        except AssertionError as e:
            print(f"Challenge 1 failed: {str(e)}")

    @staticmethod
    def inputs():
        return {
            'vocab_size': 100,  # Number of unique tokens
            'd_model': 64,  # Embedding dimension size
            'example_input': torch.randint(0, 100, (2, 5))  # Sample token indices (batch_size=2, seq_len=5)
        }


class Challenge2:
    @staticmethod
    def test(cls):
        try:
            inputs = Challenge2.inputs()
            instance = cls(inputs['d_model'], inputs['max_len'])  # Instantiate the user's class
            output = instance(inputs['example_input'])  # Run forward pass
            
            # Check if output shape matches expected shape
            assert output.shape == (2, 10, inputs['d_model']), f"Expected output shape (2, 10, {inputs['d_model']}), but got {output.shape}"
            print("Challenge 2 passed!")
        except AttributeError as e:
            print(f"Challenge 2 failed: Missing or incorrect attribute or method. {str(e)}")
        except AssertionError as e:
            print(f"Challenge 2 failed: {str(e)}")

    @staticmethod
    def inputs():
        return {
            'd_model': 64,  # Embedding dimension size
            'max_len': 10,  # Max sequence length for positional encoding
            'example_input': torch.zeros(2, 10, 64)  # Placeholder tensor (batch_size=2, seq_len=10, d_model=64)
        }


class Challenge3:
    @staticmethod
    def test(cls):
        try:
            inputs = Challenge3.inputs()
            output, attention_weights = cls(inputs['Q'], inputs['K'], inputs['V'])  # Call function with input tensors
            
            # Check if output and attention_weights shapes match expected shapes
            assert output.shape == (2, 5, inputs['Q'].size(-1)), f"Expected output shape (2, 5, {inputs['Q'].size(-1)}), but got {output.shape}"
            assert attention_weights.shape == (2, 5, 5), f"Expected attention weights shape (2, 5, 5), but got {attention_weights.shape}"
            print("Challenge 3 passed!")
        except AttributeError as e:
            print(f"Challenge 3 failed: Missing or incorrect attribute or method. {str(e)}")
        except AssertionError as e:
            print(f"Challenge 3 failed: {str(e)}")

    @staticmethod
    def inputs():
        d_k = 64
        return {
            'Q': torch.rand(2, 5, d_k),  # Query tensor (batch_size=2, seq_len=5, d_k=64)
            'K': torch.rand(2, 5, d_k),  # Key tensor (batch_size=2, seq_len=5, d_k=64)
            'V': torch.rand(2, 5, d_k)   # Value tensor (batch_size=2, seq_len=5, d_k=64)
        }


class Challenge4:
    @staticmethod
    def test(cls):
        try:
            inputs = Challenge4.inputs()
            instance = cls(inputs['d_model'], inputs['n_heads'])  # Instantiate the user's class
            output = instance(inputs['example_input'])  # Run forward pass
            
            # Check if output shape matches expected shape
            assert output.shape == (2, 10, inputs['d_model']), f"Expected output shape (2, 10, {inputs['d_model']}), but got {output.shape}"
            print("Challenge 4 passed!")
        except AttributeError as e:
            print(f"Challenge 4 failed: Missing or incorrect attribute or method. {str(e)}")
        except AssertionError as e:
            print(f"Challenge 4 failed: {str(e)}")

    @staticmethod
    def inputs():
        return {
            'd_model': 64,  # Embedding dimension size
            'n_heads': 8,   # Number of attention heads
            'example_input': torch.rand(2, 10, 64)  # Example input (batch_size=2, seq_len=10, d_model=64)
        }


class Challenge5:
    @staticmethod
    def test(cls):
        try:
            inputs = Challenge5.inputs()
            instance = cls(inputs['d_model'], inputs['d_ff'])  # Instantiate the user's class
            output = instance(inputs['example_input'])  # Run forward pass
            
            # Check if output shape matches expected shape
            assert output.shape == (2, 10, inputs['d_model']), f"Expected output shape (2, 10, {inputs['d_model']}), but got {output.shape}"
            print("Challenge 5 passed!")
        except AttributeError as e:
            print(f"Challenge 5 failed: Missing or incorrect attribute or method. {str(e)}")
        except AssertionError as e:
            print(f"Challenge 5 failed: {str(e)}")

    @staticmethod
    def inputs():
        return {
            'd_model': 64,  # Embedding dimension size
            'd_ff': 256,    # Feedforward dimension size
            'example_input': torch.rand(2, 10, 64)  # Example input (batch_size=2, seq_len=10, d_model=64)
        }

