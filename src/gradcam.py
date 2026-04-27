
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import cv2
import torchvision.transforms as transforms # Moved this import here for visualize_GradCam

class gradcam():
  def __init__(self, model, target_layer):
    self.model = model
    self.gradients = None
    self.activations = None

    target_layer.register_forward_hook(self._save_activations) # Use correct method names
    target_layer.register_backward_hook(self._save_gradients) # Use correct method names

  def _save_activations(self, module, input , output):
    """ captures forward pass feature maps """
    self.activations = output.detach()

  def _save_gradients(self, module, grad_input, grad_output):
    """ captures backward pass gradients """
    self.gradients = grad_output[0].detach()

  def generate(self,image_tensor,class_idx=None):
    """
      Generate GradCAM heatmap for given image

      Args:
          image_tensor: [1, 3, 224, 224]
          class_idx: target class (None = predicted class)

      Returns:
          heatmap: numpy array [224, 224] values 0-1
          pred_class: predicted class index
          confidence: prediction confidence
      """
    self.model.eval()

    image = image_tensor.requires_grad_(True) # Use requires_grad_ for in-place

    # Forward pass

    output = self.model(image)

    probs = F.softmax(output, dim=1)

    if class_idx is None: # Use `is None` not `in None`
      class_idx = probs.argmax().item()

    confidence = probs[0, class_idx].item()

    # backward pass

    self.model.zero_grad()

    output[0, class_idx].backward(retain_graph=True) # retain_graph=True might be needed if using model again

    # Global average pool the gradients

    weights = self.gradients.mean(dim=[2,3])[0]

    # [C, H, W] feature maps weighted by importance
    cam = torch.zeros(
        self.activations.shape[2:],
        dtype=torch.float32,
        device=self.activations.device # Ensure cam is on the same device
    )


    for i, w in enumerate(weights):
      cam += w * self.activations[0,i]

    # ReLU — keep only positive influences

    cam = F.relu(cam)


    if cam.max() > 0:
      cam = cam / cam.max()

    heatmap = cam.cpu().numpy()
    heatmap = cv2.resize(heatmap, (224,224))

    return heatmap, class_idx, confidence

# This should be a standalone function, not a method of gradcam class
def visualize_GradCam(model, dataset,
                      device,class_names,
                      num_samples=4,
                      save_path='/content/repo/results/'):

  # target_layer needs to be passed or derived correctly
  # Assuming model.backbone.features[-1] is the correct target layer for EfficientNet
  target_layer = model.backbone.features[-1]

  gc = gradcam(model, target_layer) # Instantiate gradcam class

  fig,axes = plt.subplots(
      len(class_names),
      3,
      figsize=(12,4 *len(class_names))
  )

  fig.suptitle(
      'GradCAM-regions influenced prediction-',
      fontsize=14,fontweight='bold'

  )

  col_titles = ['Original MRI','GradCAM Heatmap','Overlay']

  for ax, title in zip(axes[0],col_titles):
    ax.set_title(title,fontweight='bold',fontsize=11)

  val_transform = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=[0.485, 0.456, 0.406],
          std=[0.229, 0.224, 0.225 ])


  ])

  for class_idx, class_name in enumerate(class_names): # Iterate through class_names correctly
    class_samples = [s for s in dataset.samples if s[1] == class_idx]
    path, label = class_samples[0] # Take the first sample for visualization

    img_pil = Image.open(path).convert('RGB')
    img_pil = img_pil.resize((224,224))
    img_np = np.array(img_pil)


    # prepare tensor

    img_tensor = val_transform(
      Image.open(path).convert('RGB')).unsqueeze(0).to(device)

    # Generate GradCAM

    heatmap, pred_idx, confidence = gc.generate( # Call generate on the instance gc
        img_tensor, class_idx=class_idx
    )

    # create colored heatmap
    heatmap_colored = cm.jet(heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    # create overlay

    overlay = (0.6 * img_np + 0.4 * heatmap_colored).astype(np.uint8)

    # Plot

    axes[class_idx][0].imshow(img_np, cmap='gray')
    axes[class_idx][0].set_ylabel(
      f'{class_name}\n(true)',
      fontsize=11, fontweight='bold'

    )
    axes[class_idx][0].axis('off')

    axes[class_idx][1].imshow(heatmap, cmap='jet')
    axes[class_idx][1].set_title(
        f'pred: {class_names[pred_idx]} ' # Use class_names for prediction
        f'({confidence:.1%})',
        fontsize=9
    )
    axes[class_idx][1].axis('off')

    axes[class_idx][2].imshow(overlay)
    axes[class_idx][2].axis('off')

  plt.tight_layout()
  plt.savefig(f'{save_path}/gradcam_results.png',
              dpi=150, bbox_inches='tight')
  plt.show()
  print("GradCAM visualization saved ")
