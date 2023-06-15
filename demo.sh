
imgPath="dataset/test/"
imgs=()
for file in 000000000285.jpg  000000000632.jpg  000000000724.jpg  000000109477.jpg 000000581781.jpg
do
    imgs+="${imgPath}${file} "
done
echo $imgs
classes_list=()
for class in banana bed light bear person other grass
do
    classes_list+="${class} "
done
echo ${classes_list}
python3 demo.py --input ${imgs}  --output ./output2/ #--class-names ${classes_list}
