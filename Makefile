.PHONY: all

all: AMAZON BBC OHSUMED Reuters-21578 RCV1v2 CRANFIELD 20NEWS TWITTER NTCIR corpora vectors matrices

AMAZON:
	mkdir -p $@
	cd $@ && curl http://jmcauley.ucsd.edu/data/amazon/ | xmllint -html -xpath //a/@href - 2>/dev/null | sed -r 's/ href="([^"]*)"/\1\n/g' | grep -F '_5.json.gz' | parallel --bar --halt=2 wget
	parallel --bar --halt=2 -- 'gzip -d <{} | split -C 20m --numeric-suffixes - {}_split' ::: AMAZON/*.json.gz

BBC:
	mkdir -p $@
	cd $@ && wget http://mlg.ucd.ie/files/datasets/bbc.zip
	cd $@ && wget http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip
	cd $@ && wget http://mlg.ucd.ie/files/datasets/bbcsport.zip
	cd $@ && wget http://mlg.ucd.ie/files/datasets/bbcsport-fulltext.zip
	cd $@ && parallel --halt=2 -- unzip ::: *.zip
	recode latin2..utf8 $@/bbc/*/*.txt $@/bbcsport/*/*.txt

OHSUMED:
	mkdir -p $@
	cd $@ && wget http://disi.unitn.it/moschitti/corpora/ohsumed-all-docs.tar.gz
	cd $@ && wget http://disi.unitn.it/moschitti/corpora/First-Level-Categories-of-Cardiovascular-Disease.txt
	cd $@ && tar xzvf ohsumed-all-docs.tar.gz

Reuters-21578:
	mkdir -p $@
	cd $@ && wget http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz

RCV1v2:
	mkdir -p $@
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a01-list-of-topics/rcv1.topics.txt
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a02-orig-topics-hierarchy/rcv1.topics.hier.orig
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a03-expanded-topics-hierarchy/rcv1.topics.hier.expanded
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a04-list-of-industries/rcv1.industries.txt
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a05-industry-hierarchy/rcv1.industries.hier
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a06-list-of-regions/rcv1.regions.txt
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a07-rcv1-doc-ids/rcv1v2-ids.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a09-industry-qrels/rcv1-v2.industries.qrels.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a10-region-qrels/rcv1-v2.regions.qrels.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a11-smart-stop-list/english.stop
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt0.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt1.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt2.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_test_pt3.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004_tokens_train.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004-non-v2_tokens_test_pt0.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004-non-v2_tokens_test_pt1.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004-non-v2_tokens_test_pt2.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004-non-v2_tokens_test_pt3.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a12-token-files/lyrl2004-non-v2_tokens_train.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt0.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt1.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt2.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_test_pt3.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004_vectors_train.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004-non-v2_vectors_test_pt0.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004-non-v2_vectors_test_pt1.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004-non-v2_vectors_test_pt2.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004-non-v2_vectors_test_pt3.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a13-vector-files/lyrl2004-non-v2_vectors_train.dat.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a14-term-dictionary/stem.termid.idf.map.txt
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a15-contingency-tables/a15-contingency-tables.tar.gz
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a16-rbb-topic/topics.rbb
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a17-rbb-industry/industries.rbb
	cd $@ && wget http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a18-rbb-region/regions.rbb

CRANFIELD:
	mkdir -p $@
	cd $@ && wget https://web.archive.org/web/20180403011709/http://ir.dcs.gla.ac.uk/resources/test_collections/cran/cran.tar.gz

20NEWS:
	mkdir -p $@
	cd $@ && wget http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz
	cd $@ && wget http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
	cd $@ && wget http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz

TWITTER:
	mkdir -p $@
	cd $@ && wget https://web.archive.org/web/20180328044709/www.sananalytics.com/lab/twitter-sentiment/sanders-twitter-0.2.zip
	cd $@ && wget https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv

NTCIR:
	ln -s /mnt/storage/ntcir $@

corpora:
	mkdir -p $@
	cd $@ && wget http://mattmahoney.net/dc/enwik8.zip
	cd $@ && parallel --halt=2 -- unzip ::: *.zip
	cd $@ && perl ../wikifil.pl enwik8 > fil8

matrices:
	mkdir -p $@

figures:
	mkdir -p $@

vectors:
	make Word2Bits
	make corpora
	mkdir -p $@
	Word2Bits/word2bits -sample 1e-4 -bitlevel 0 -size 200  -window 10 -negative 24 -threads $(shell nproc) -iter 10 -min-count 5 -train corpora/fil8 -output vectors/32b_200d_vectors_e10_nonbin -binary 0
	Word2Bits/word2bits -sample 1e-4 -bitlevel 1 -size 1000 -window 10 -negative 24 -threads $(shell nproc) -iter 10 -min-count 5 -train corpora/fil8 -output vectors/1b_1000d_vectors_e10_nonbin -binary 0

Word2Bits:
	git clone https://github.com/agnusmaximus/Word2Bits
	make -C $@ word2bits
