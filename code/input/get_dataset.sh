# Path: code/input/get_dataset.sh

if [ -z "$1" ]; then
    set -- "v2"
fi
if [ "$1" == "v2" ]; then
    wget -O data.zip https://scontent.fcgh22-1.fna.fbcdn.net/m1/v/t6/An-ZjB1Zm61yAZG0ozTymz8I8NqI4x0MrYrh26dq7kPgfu8vf9ImrdaOAVOFYbJ2pNAgUnVGBmbue9lTgdBOb5BbKXIpFs0fpYWqACbrQDChAA2fdX0zS9PcHu7fY8c-FOvyBVxPNYNFQuM.zip?ccb=10-5&oh=00_AfDO4XcGZdP2Sz1zLrvZgjE7GcFNJPOTohkezA6X6XiGDg&oe=6541C965&_nc_sid=6de079
    unzip data.zip
    rm data.zip
    exit 0
fi
if [ "$1" == "v1.2" ]; then
    wget -O data.zip https://scontent.fcgh22-1.fna.fbcdn.net/m1/v/t6/An_o5cmHOsS1VbLdaKx_zfMdi0No5LUpL2htRxMwCjY_bophtOkM0-6yTKB2T2sa0yo1oP086sqiaCjmNEw5d_pofWyaE9LysYJagH8yXw_GZPzK2wfiQ9u4uAKrVcEIrkJiVuTn7JBumrA.zip?ccb=10-5&oh=00_AfB-aW_RPKJGPEa0yh-joqWA5rUZDIVb_pxdl9c06uSUWA&oe=6541E968&_nc_sid=6de079
    unzip data.zip
    rm data.zip
    exit 0
fi
