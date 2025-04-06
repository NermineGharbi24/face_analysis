# main.py
import os
import argparse
from face_analysis import FaceAnalyzer

def ensure_model_files():
    """
    Check for face detection model files and download them if not present.
    """
    model_dir = 'models'
    prototxt_path = os.path.join(model_dir, 'deploy.prototxt')
    caffemodel_path = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    # Check if files exist
    if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
        print("Downloading face detection model files...")
        import urllib.request
        
        # URLs for the model files
        prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        # Download files
        urllib.request.urlretrieve(prototxt_url, prototxt_path)
        urllib.request.urlretrieve(caffemodel_url, caffemodel_path)
        
        print("Model files downloaded successfully.")

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Face Analysis Tool')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Image processing command
    image_parser = subparsers.add_parser('image', help='Process an image')
    image_parser.add_argument('--input', type=str, required=True, help='Path to input image')
    image_parser.add_argument('--output', type=str, help='Path to save output image')
    image_parser.add_argument('--model', type=str, default='models/best_model.pth', 
                             help='Path to the trained model')
    image_parser.add_argument('--show', action='store_true', help='Show the result')
    
    # Video processing command
    video_parser = subparsers.add_parser('video', help='Process a video')
    video_parser.add_argument('--input', type=str, required=True, help='Path to input video')
    video_parser.add_argument('--output', type=str, help='Path to save output video')
    video_parser.add_argument('--model', type=str, default='models/best_model.pth', 
                             help='Path to the trained model')
    
    return parser.parse_args()

def main():
    """
    Main function to run the face analysis tool.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Ensure model files are present
    ensure_model_files()
    
    # Create face analyzer
    analyzer = FaceAnalyzer(model_path=args.model if hasattr(args, 'model') else None)
    
    # Process based on command
    if args.command == 'image':
        print(f"Processing image: {args.input}")
        results = analyzer.process_image(
            image_path=args.input,
            output_path=args.output,
            show_result=args.show
        )
        print(f"Detected {len(results)} faces.")
        for i, face in enumerate(results):
            print(f"Face {i+1}:")
            print(f"  - Gender: {face['predictions']['gender']}")
            print(f"  - Age: {face['predictions']['age']}")
            print(f"  - Race: {face['predictions']['race']}")
        
    elif args.command == 'video':
        print(f"Processing video: {args.input}")
        analyzer.process_video(
            video_path=args.input,
            output_path=args.output
        )
        print("Video processing completed.")
    
    else:
        print("Please specify a command: 'image' or 'video'")
        
if __name__ == "__main__":
    main()
