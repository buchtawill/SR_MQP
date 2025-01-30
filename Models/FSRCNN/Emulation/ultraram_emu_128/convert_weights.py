import torch
import os

class FSRCNN(torch.nn.Module):
    def __init__(self):
        super(FSRCNN, self).__init__()
        
        # Load the state dict first to get the shapes
        weights_path = '../100E_5em4_b64_CPU.pth'
        state_dict = torch.load(weights_path, map_location='cpu')
        
        # Define layers with exact same names as in state dict
        self.feature_extraction = torch.nn.Sequential(
            torch.nn.Conv2d(3, 56, kernel_size=5, padding=2),  # .0
            torch.nn.PReLU(56)  # .1
        )
        
        self.shrink = torch.nn.Sequential(
            torch.nn.Conv2d(56, 12, kernel_size=1),  # .0
            torch.nn.PReLU(12)  # .1
        )
        
        # Map part (using exact same structure as weights)
        self.map = torch.nn.Sequential(
            torch.nn.Conv2d(12, 12, kernel_size=3, padding=1),  # .0
            torch.nn.PReLU(12),  # .1
            torch.nn.Conv2d(12, 12, kernel_size=3, padding=1),  # .2
            torch.nn.PReLU(12),  # .3
            torch.nn.Conv2d(12, 12, kernel_size=3, padding=1),  # .4
            torch.nn.PReLU(12),  # .5
            torch.nn.Conv2d(12, 12, kernel_size=3, padding=1),  # .6
            torch.nn.PReLU(12)   # .7
        )
        
        self.expand = torch.nn.Sequential(
            torch.nn.Conv2d(12, 56, kernel_size=1),  # .0
            torch.nn.PReLU(56)  # .1
        )
        
        self.deconv = torch.nn.ConvTranspose2d(56, 3, kernel_size=9, stride=2, padding=4, output_padding=1)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        x = self.deconv(x)
        return x

def print_model_state_dict(model):
    print("\nModel state dict keys:")
    for key in model.state_dict().keys():
        print(f"  {key}")

def convert_weights_to_torchscript():
    print("Starting conversion process...")
    
    # Create model
    print("Creating FSRCNN model...")
    model = FSRCNN()
    
    # Print model's expected keys
    print_model_state_dict(model)
    
    # Load weights
    print("\nLoading weights...")
    weights_path = '../100E_5em4_b64_CPU.pth'
    state_dict = torch.load(weights_path, map_location='cpu')
    
    print("\nWeight file keys:")
    for key in state_dict.keys():
        print(f"  {key}")
    
    # Load weights into model
    print("\nLoading weights into model...")
    model.load_state_dict(state_dict)
    print("Successfully loaded weights")
    
    # Set to eval mode
    model.eval()
    
    # Test the model
    print("\nTesting model...")
    with torch.no_grad():
        test_input = torch.randn(1, 3, 32, 32)
        test_output = model(test_input)
        print(f"Test input shape: {test_input.shape}")
        print(f"Test output shape: {test_output.shape}")
    
    # Convert to TorchScript
    print("\nConverting to TorchScript...")
    scripted_model = torch.jit.trace(model, test_input)
    
    # Save the model
    print("\nSaving model...")
    scripted_model.save('fsrcnn_model.pt')
    print("Saved TorchScript model to fsrcnn_model.pt")
    
    # Verify saved model
    print("\nVerifying saved model...")
    loaded_model = torch.jit.load('fsrcnn_model.pt')
    loaded_model.eval()
    with torch.no_grad():
        verify_output = loaded_model(test_input)
        print(f"Verification output shape: {verify_output.shape}")
    
    print("\nConversion completed successfully!")

if __name__ == "__main__":
    try:
        convert_weights_to_torchscript()
    except Exception as e:
        print(f"Error during conversion: {str(e)}")