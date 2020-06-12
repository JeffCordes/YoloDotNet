using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Threading.Tasks;
using YoloDotNetApi.Models;

namespace YoloDotNetApi.Utilities
{
    public static class ImageUtils
    {
        /// <summary>
        /// Resize an image to the specified height and width with letterboxing
        /// </summary>
        /// <param name="bit">source image</param>
        /// <param name="width">width</param>
        /// <param name="height">height</param>
        /// <returns>the resized bitmap image</returns>
        public static Bitmap ResizeImage(Image bit, int width, int height)
        {

            //original stream reader writes all horz, so check exif
            foreach (PropertyItem p in bit.PropertyItems)
            {
                if (p.Id == 274)
                {
                    int orientation = (int)p.Value[0];
                    if (orientation == 6)
                        bit.RotateFlip(RotateFlipType.Rotate90FlipNone);
                    if (orientation == 8)
                        bit.RotateFlip(RotateFlipType.Rotate270FlipNone);
                    break;
                }
            }

            int nW = width;
            int nH = height;

            if (bit.Width > bit.Height)
            {
                decimal ratio = decimal.Divide(Convert.ToDecimal(bit.Height), Convert.ToDecimal(bit.Width));
                nH = (int)decimal.Multiply(nW, ratio);
            }
            else
            {
                decimal ratio = decimal.Divide(Convert.ToDecimal(bit.Width), Convert.ToDecimal(bit.Height));
                nW = (int)decimal.Multiply(nH, ratio);
            }
            Bitmap canvas = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);

            int offsetX = 0;
            int offsetY = 0;

            if (nW < width)
            {
                offsetX = (width - nW) / 2;
            }

            if (nH < height)
            {
                offsetY = (height - nH) / 2;
            }

            using (Graphics g = Graphics.FromImage((Image)canvas))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                g.DrawImage(bit, offsetX, offsetY, nW, nH);
            }
            return canvas;
        }

        /// <summary>
        /// Get a normalized float array from an image,  this is the yolo input. RB is reversed, and values are between 0 and 1
        /// this is there relative value between 0 and 255
        /// </summary>
        /// <param name="bit">source bitmap</param>
        /// <returns>float array</returns>
        public static float[] GetNormalizedFloatArray(Bitmap bit)
        {
            unsafe
            {
                BitmapData bitmapData = bit.LockBits(new Rectangle(0, 0, bit.Width, bit.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);

                int bytesPerPixel = 3;
                int heightInPixels = bitmapData.Height;
                int widthInBytes = bitmapData.Width * bytesPerPixel;
                byte* PtrFirstPixel = (byte*)bitmapData.Scan0;

                float[] floatData = new float[(bit.Width * bit.Height) * bytesPerPixel];

                var floatsInOneLine = bit.Width * bytesPerPixel;

                Parallel.For(0, heightInPixels, y =>
                {
                    //var lineStartIndex = y * floatsInOneLine;
                    var lineStartIndex = y * 416;

                    byte* currentLine = PtrFirstPixel + (y * bitmapData.Stride);
                    for (int x = 0; x < widthInBytes; x = x + bytesPerPixel)
                    {
                        var pixelIndex = lineStartIndex + (x / 3);

                        floatData[pixelIndex] = (float)decimal.Divide((decimal)currentLine[x + 2], (decimal)255);
                        floatData[pixelIndex + 173056] = (float)decimal.Divide((decimal)currentLine[x + 1], (decimal)255);
                        floatData[pixelIndex + 346112] = (float)decimal.Divide((decimal)currentLine[x + 0], (decimal)255);
                    }
                });
                bit.UnlockBits(bitmapData);

                return floatData;
            }
        }

        /// <summary>
        /// Draw yolo boxes on an image
        /// </summary>
        /// <param name="image">the image to draw on</param>
        /// <param name="predictions">a list of yolo predictions</param>
        /// <returns>the image with boxes drawn on it</returns>
        public static Image DrawYoloBoxes(Image image, List<YoloPrediction> predictions)
        {
            var origWidth = image.Width;
            var origHeight = image.Height;

            var largestDim = origWidth;
            if (origHeight > origWidth) largestDim = origHeight;

            // draw a box for each prediction
            foreach (var prediction in predictions)
            {
                // filter boundaries outside of the image
                var x = (int)Math.Max(prediction.Box.X, 0);
                var y = (int)Math.Max(prediction.Box.Y, 0);
                var width = (int)Math.Min(largestDim - x, prediction.Box.Width);
                var height = (int)Math.Min(largestDim - y, prediction.Box.Height);

                // scale from the yoloy input size
                x = (int)decimal.Multiply((decimal)x, decimal.Divide(largestDim, 416));
                y = (int)decimal.Multiply((decimal)y, decimal.Divide(largestDim, 416));
                width = (int)decimal.Multiply((decimal)width, decimal.Divide(largestDim, 416));
                height = (int)decimal.Multiply((decimal)height, decimal.Divide(largestDim, 416));

                // Adjust for letterbox offset
                if (origHeight > origWidth)
                {
                    x = x - ((origHeight - origWidth) / 2);
                }
                else if (origWidth > origHeight)
                {
                    y = y - ((origWidth - origHeight) / 2);
                }

                // ensure edges do not go off edge of image
                if (x < 0) x = 0;
                if (x > origWidth) x = origWidth;
                if (y < 0) y = 0;
                if (y > origHeight) y = origHeight;
                if (x + width > origWidth) width = origWidth - x;
                if (y + height > origHeight) height = origHeight - y;

                var labelToDraw = $"{prediction.TopPrediction.Label} ({(int)(prediction.ContainsObjectConfidence * 100)})%";
                using (Graphics gfx = Graphics.FromImage(image))
                {
                    gfx.CompositingQuality = CompositingQuality.HighQuality;
                    gfx.SmoothingMode = SmoothingMode.HighQuality;
                    gfx.InterpolationMode = InterpolationMode.HighQualityBicubic;

                    // Define Text Options
                    Font drawFont = new Font("Arial", 12, FontStyle.Bold);
                    SizeF size = gfx.MeasureString(labelToDraw, drawFont);
                    SolidBrush fontBrush = new SolidBrush(Color.Black);
                    Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

                    var colorToDraw = GetRandomColor();
                    Pen pen = new Pen(colorToDraw, 3.2f);
                    SolidBrush colorBrush = new SolidBrush(colorToDraw);

                    // Draw text on image 
                    gfx.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
                    gfx.DrawString(labelToDraw, drawFont, fontBrush, atPoint);

                    // Draw bounding box on image
                    gfx.DrawRectangle(pen, x, y, width, height);
                }
            }
            return image;
        }

        /// <summary>
        /// Gets a random color
        /// </summary>
        /// <returns>generated color</returns>
        public static Color GetRandomColor()
        {
            var random = new Random();
            return Color.FromArgb(random.Next(0, 255), random.Next(0, 255), random.Next(0, 255));
        }
    }

}
