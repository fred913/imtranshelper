import cv2
import typer

from imtranshelper import ocr_and_mark


def main(image_path: str = typer.Argument(...,
                                          help="Path to the input image."),
         output_path: str | None = typer.Argument(
             None, help="Path to the output image with marked text.")):
    masked_im, text_list = ocr_and_mark(image_path)
    print(text_list)
    if output_path:
        cv2.imwrite(output_path, masked_im)
        print(f"Marked image saved to {output_path}")
    else:
        cv2.imshow("marked_image", masked_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


typer.run(main)
