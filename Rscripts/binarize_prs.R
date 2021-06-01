require(data.table)

binarize_prs_gpc <-
  function(file_tr,
           file_ts,
           ancestry_tr,
           ancestry_ts,
           outfile) {
    data_tr <- fread(file_tr, sep = '\t')
    data_ts <- fread(file_ts, sep = '\t')

    anc_tr <- fread(ancestry_tr)
    anc_ts <- fread(ancestry_ts)

    data_tr$ancestry <- anc_tr$ancestry
    data_ts$ancestry <- anc_ts$ancestry

    new_data_tr <- data.table()
    new_data_ts <- data.table()
    for (a in c("eur", "afr", "amr")) {
      tmp_data_tr <- data_tr[ancestry == a]
      tmp_data_ts <- data_ts[ancestry == a]
      top_dec <-
        quantile(tmp_data_tr[, pgcscz2_gpc.p5e.1],
                 prob = seq(0, 1, length = 11),
                 type = 5)["90%"]
      tmp_data_tr$pgcscz2_gpc.p5e.1[which(tmp_data_tr$pgcscz2_gpc.p5e.1 < top_dec)] <-
        0
      tmp_data_tr$pgcscz2_gpc.p5e.1[which(tmp_data_tr$pgcscz2_gpc.p5e.1 >= top_dec)] <-
        1
      tmp_data_ts$pgcscz2_gpc.p5e.1[which(tmp_data_ts$pgcscz2_gpc.p5e.1 < top_dec)] <-
        0
      tmp_data_ts$pgcscz2_gpc.p5e.1[which(tmp_data_ts$pgcscz2_gpc.p5e.1 >= top_dec)] <-
        1
      new_data_tr <- rbind(new_data_tr, tmp_data_tr)
      new_data_ts <- rbind(new_data_ts, tmp_data_ts)
    }

    fwrite(new_data_tr, sprintf("%s_train.txt", outfile), sep = '\t')
    fwrite(new_data_ts, sprintf("%s_test.txt", outfile), sep = '\t')
  }


binarize_prs_biome <-
  function(file_tr,
           file_ts,
           outfile) {
    data_tr <- fread(file_tr, sep = '\t')
    data_ts <- fread(file_ts, sep = '\t')
    
    new_data_tr <- data.table()
    new_data_ts <- data.table()
    for (a in c("EUR", "AFR", "AMR")) {
      tmp_data_tr <- data_tr[gill.ContinentalGrouping == a]
      tmp_data_ts <- data_ts[gill.ContinentalGrouping == a]
      top_dec <-
        quantile(tmp_data_tr[, PRS],
                 prob = seq(0, 1, length = 11),
                 type = 5)["90%"]
      tmp_data_tr$PRS[which(tmp_data_tr$PRS < top_dec)] <-
        0
      tmp_data_tr$PRS[which(tmp_data_tr$PRS >= top_dec)] <-
        1
      tmp_data_ts$PRS[which(tmp_data_ts$PRS < top_dec)] <-
        0
      tmp_data_ts$PRS[which(tmp_data_ts$PRS >= top_dec)] <-
        1
      new_data_tr <- rbind(new_data_tr, tmp_data_tr)
      new_data_ts <- rbind(new_data_ts, tmp_data_ts)
    }

    fwrite(new_data_tr, sprintf("%s_train.txt", outfile), sep = '\t')
    fwrite(new_data_ts, sprintf("%s_test.txt", outfile), sep = '\t')
  }

# BIOME
binarize_prs_biome('../data/biome_all_agressive_train.txt',
                   '../data/biome_all_agressive_test.txt',
                   '../data/biome_all_agressive_binarizedprs')
binarize_prs_biome('../data/biome_genetic_agressive_train.txt',
                   '../data/biome_genetic_agressive_test.txt',
                   '../data/biome_genetic_agressive_binarizedprs')

binarize_prs_biome('../data/biome_all_psych_admit_train.txt',
                   '../data/biome_all_psych_admit_test.txt',
                   '../data/biome_all_psych_admit_binarizedprs')
binarize_prs_biome('../data/biome_genetic_psych_admit_train.txt',
                   '../data/biome_genetic_psych_admit_test.txt',
                   '../data/biome_genetic_psych_admit_binarizedprs')


# GPC
binarize_prs_gpc('../data/gpc_all_opcrit90_train.txt',
                   '../data/gpc_all_opcrit90_test.txt',
                   '../data/gpc_all_opcrit90_binarizedprs')
binarize_prs_biome('../data/gpc_genetic_opcrit90_train.txt',
                   '../data/gpc_genetic_opcrit90_test.txt',
                   '../data/gpc_genetic_opcrit90_binarizedprs')


