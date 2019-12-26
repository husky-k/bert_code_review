wget http://opus.nlpl.eu/download.php?f=OpenSubtitles/v2016/mono/OpenSubtitles.raw.en.gz -O dataset.txt.gz
gzip -d dataset.txt.gz
tail dataset.txt

CORPUS_SIZE=10000
(head -n $CORPUS_SIZE dataset.txt) > subdataset.txt
mv subdataset.txt dataset.txt

