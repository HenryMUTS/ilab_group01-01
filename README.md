<img width="887" height="742" alt="image" src="https://github.com/user-attachments/assets/d66c3358-bbcc-4112-8efa-97b1089fa6a5" />

<img width="865" height="421" alt="image" src="https://github.com/user-attachments/assets/7d0e3b43-cb1a-41ac-be77-635ad8a10500" />

Last step run following command with your prediction images inside codeformer/inputs/whole_imgs
you can decide the output path:

python inference_codeformer.py \
  -i inputs/whole_imgs\
  -o Final_result\
  -w 1\
  --face_upsample \
  --bg_upsampler realesrgan \
  --bg_tile 400 \
  -s 2

After this we have to seprately run merge.py where u have to manually change the input/output paths and mask path. then simply run it.

<img width="619" height="194" alt="image" src="https://github.com/user-attachments/assets/f5b51f3c-e0a7-41d8-a455-fa3baf09fcf1" />

  
