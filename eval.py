import torch
from torchvision import transforms
from PIL import Image

from data import EmotionEncoding

classes = ['angry', 'fear', 'happy', 'sad', 'surprise']
model = EmotionEncoding()
model.load_state_dict(torch.load('best_emotion_model.pth', map_location='cpu'))
model.eval()

img = Image.open('IMG_1413.jpeg')

transform = transforms.Compose([
    transforms.Resize((28, 28)),   
    transforms.ToTensor(),
])

img_tensor = transform(img).unsqueeze(0)   

with torch.no_grad():
    output = model(img_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    print("Predicted:", classes[predicted_class.item()])
    print(probabilities)
