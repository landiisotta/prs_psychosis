require(data.table)
require(caret)
require(ROCR)
require(nnet)

set.seed(42)

# Recode function
recode_fun <- function(df, vect, name) {
  d_rid <- desc[item == name, c('categoricalValue', 'numericValue')]
  for (cat in unique(d_rid$categoricalValue)) {
    if (name == 'DIPAD.PSYCHOTICSX2WWITHOUTMOODSX') {
      vect[vect == "1=Yes"] <- 1
      vect[vect == "2=No"] <- 0
    } else{
      if (name == "OPCRIT.89") {
        vect[grep("No response", vect)] <- 0
        vect[grep("Positive response", vect)] <- 1
      } else{
        vect[vect == cat] <- d_rid[categoricalValue == cat, numericValue]
      }
    }
  }
  return(as.factor(vect))
}


# Imputation function
impute_data <- function(train, test, c_feat, cat_feat) {
  for (k in names(train)) {
    if (k %in% c_feat) {
      # impute numeric variables with median
      med <- median(train[[k]], na.rm = T)
      set(x = train, which(is.na(train[[k]])), k, med)
      set(x = test, which(is.na(test[[k]])), k, med)
      
    } else if (k %in% cat_feat) {
      ## impute categorical variables with mode
      mode <- names(which.max(table(train[[k]])))
      set(x = train, which(is.na(train[[k]])), k, mode)
      set(x = test, which(is.na(test[[k]])), k, mode)
    }
  }
  return(list(train, test))
}


##############
## Run code ##
##############

# Read datasets
# Dataset description file
desc <-
  fread('../data/opcrit_helper_2020.txt',
        sep = '|',
        header = TRUE)

# Dataset
gpc <-
  fread('../data/gpc.tsv',
        sep = '\t',
        header = TRUE)

# Data types
gpc_desc <-
  fread('../data/gpc_coltypes.tsv',
        sep = '\t',
        header = TRUE)

# Initialize variable names (clinical, prs, principal components)
clin_vars <- gpc_desc[group == "OPCRIT" | group == "DIPAD", variable]
pc_vars <- gpc_desc[group == "Genetic_PCs", variable]
prs_var <- "pgcscz2_gpc.p5e.1"

# Recode variables
# Types
c_items <- unique(desc[class == 'continuous', item])
bc_items <- unique(desc[class == 'binary_categorical', item])
oc_items <- unique(desc[class == 'ordered_categorical', item])
cat_items <- unique(desc[class == 'categorical', item])
n_items <- unique(desc[class == 'numeric', item])

# Recode
rc_feat <- c(bc_items, cat_items, oc_items)
# Add continuous features
rc_gpc <- gpc[, c(c_items, n_items), with = FALSE]

for (nfeat in rc_feat) {
  if (nfeat %in% oc_items) {
    tmp_v <- recode_fun(desc, gpc[, get(nfeat)], nfeat)
    rc_gpc <- rc_gpc[, `:=`(c(nfeat), factor(tmp_v, ordered = TRUE))]
  } else{
    tmp_v <- recode_fun(desc, gpc[, get(nfeat)], nfeat)
    rc_gpc <- rc_gpc[, `:=`(c(nfeat), tmp_v)]
  }
}

# Exclude scores according to skipping rules
sk_feat <- unique(desc[skippingRules != "", item])

for (feat in sk_feat) {
  skr <- unlist(strsplit(desc[item == feat, skippingRules], '\\.'))
  skr_f <- skr[grep("OPCRIT", skr)]
  if (length(skr_f) > 0) {
    s1 <- unlist(strsplit(skr_f[1], ""))
    feat1 <-
      paste(paste(s1[1:6], collapse = ""), paste(s1[7:8], collapse = ""), sep =
              ".")
    s2 <- unlist(strsplit(skr_f[2], ""))
    feat2 <-
      paste(paste(s2[1:6], collapse = ""), paste(s2[7:8], collapse = ""), sep =
              ".")
    rc_gpc[get(feat1) == 0 & get(feat2) == 0, feat] <- NA
  } else{
    rc_gpc[DIPAD.DELUSIONS == 0, feat] <- "0"
  }
}

gpc_new <- as.data.table(cbind(rc_gpc, gpc[,-names(rc_gpc), with = FALSE]))

# Response variable OPCRIT.90 (1,2,3,4-->0; 5-->1 recoding)
gpc_opcrit90 <- gpc_new[!is.na(OPCRIT.90)]
gpc_opcrit90[, OPCRIT.90:=as.integer(OPCRIT.90)][OPCRIT.90 <= 4, OPCRIT.90:=0]
gpc_opcrit90[OPCRIT.90 == 5, OPCRIT.90:=1]
gpc_opcrit90[,OPCRIT.90:=as.factor(OPCRIT.90)]

# Generate ordered clinical features with outcome variable last
clin_vars_90 <- c()
for (cv in clin_vars){
  if (cv != 'DIPAD.AGE'){
    if (cv != "OPCRIT.90"){
    clin_vars_90 <- c(clin_vars_90, cv)}
  }
}
clin_vars_90 <- sort(c(clin_vars_90, "OPCRIT.90"))
clin_vars_90 <- c(clin_vars_90)

# Create complete datasets of desired features
feat90 <- c(clin_vars_90[1:(length(clin_vars_90)-1)], 
            pc_vars, prs_var, 'ancestry', "OPCRIT.90")
opcrit90_all <- gpc_opcrit90[, ..feat90]

# Train and test splits stratified according to: 
# ancestry, outcome
strat_90 <- paste(gpc_opcrit90$ancestry,
                  gpc_opcrit90$OPCRIT.90, sep='-')
train.index <- createDataPartition(strat_90, p=.7, list = FALSE)

train_90 <- opcrit90_all[train.index]
test_90 <- opcrit90_all[-train.index]


# Check feature-wise NA thresholds for clinical variables (PCs and PRS are complete)
fw_vect_clin_tr90 <- apply(train_90[, ..clin_vars_90], 
                           MARGIN = 2, function(x) sum(is.na(x))/length(x))
drop_feat_90 <- names(fw_vect_clin_tr90[fw_vect_clin_tr90>0.7])
drop_feat_90

fw_vect_clin_ts90 <- apply(test_90[, ..clin_vars_90], 
                           MARGIN = 2, function(x) sum(is.na(x))/length(x))
names(fw_vect_clin_ts90[fw_vect_clin_ts90>0.7])

# Drop features wit NA counts > 0.7 and impute categorical variables with mode
# and continuous variables with median (from training)
train_90 <- train_90[, -c("OPCRIT.08", "OPCRIT.19", "OPCRIT.20", "OPCRIT.21", 
                          "OPCRIT.22", "OPCRIT.22a", "OPCRIT.22b", 
                          "OPCRIT.30", "OPCRIT.31", "OPCRIT.53", "OPCRIT.56")]
test_90 <- test_90[, -c("OPCRIT.08", "OPCRIT.19", "OPCRIT.20", "OPCRIT.21", 
                        "OPCRIT.22", "OPCRIT.22a", "OPCRIT.22b", 
                        "OPCRIT.30", "OPCRIT.31", "OPCRIT.53", "OPCRIT.56")]

eur90 <- impute_data(train_90[ancestry=='eur'], test_90[ancestry=='eur'], 
            c(c_items, n_items), rc_feat)
afr90 <- impute_data(train_90[ancestry=='afr'], test_90[ancestry=='afr'], 
            c(c_items, n_items), rc_feat)
amr90 <- impute_data(train_90[ancestry=='amr'], test_90[ancestry=='amr'], 
            c(c_items, n_items), rc_feat)

out_90 <- list(rbind(eur90[[1]], afr90[[1]], amr90[[1]]),
               rbind(eur90[[2]], afr90[[2]], amr90[[2]]))

# Prepare datasets CLINICAL; CLINICAL+PCs+PRS; PCs+PRS
# Remove FID and ancestry and create dummy variables for categorical features (in cat_items)
o90_tr <- out_90[[1]]$OPCRIT.90
out_90[[1]] <- fastDummies::dummy_cols(out_90[[1]][, -"OPCRIT.90"], 
                                       select_columns = cat_items, 
                                       remove_selected_columns = TRUE, 
                                       remove_first_dummy = TRUE)

o90_ts <- out_90[[2]]$OPCRIT.90
out_90[[2]] <- fastDummies::dummy_cols(out_90[[2]][, -"OPCRIT.90"], 
                                       select_columns = cat_items, 
                                       remove_selected_columns = TRUE, 
                                       remove_first_dummy = TRUE)

cols <- c(sort(c(clin_vars_90[which(clin_vars_90 %in% names(out_90[[1]]))], 
          names(out_90[[1]])[grep("_[0-9]", names(out_90[[1]]))])), "ancestry")

clinical90_tr <- out_90[[1]][, ..cols]
clinical90_tr$OPCRIT.90 <- o90_tr
clinical90_ts <- out_90[[2]][, ..cols]
clinical90_ts$OPCRIT.90 <- o90_ts

fullcols <- c(cols[-grep('ancestry', cols)], pc_vars, prs_var, 'ancestry')
genetic90_tr <- out_90[[1]][, ..fullcols]
genetic90_tr$OPCRIT.90 <- o90_tr
genetic90_ts <- out_90[[2]][, ..fullcols]
genetic90_ts$OPCRIT.90 <- o90_ts

ridcols <- c(pc_vars, prs_var, "ancestry")
onlygen90_tr <- out_90[[1]][, ..ridcols]
onlygen90_tr$OPCRIT.90 <- o90_tr
onlygen90_ts <- out_90[[2]][, ..ridcols]
onlygen90_ts$OPCRIT.90 <- o90_ts

fwrite(clinical90_tr, '../out/gpc_clinical_opcrit90_train.txt', sep='\t')
fwrite(clinical90_ts, '../out/gpc_clinical_opcrit90_test.txt', sep='\t')

fwrite(genetic90_tr, '../out/gpc_all_opcrit90_train.txt', sep='\t')
fwrite(genetic90_ts, '../out/gpc_all_opcrit90_test.txt', sep='\t')

fwrite(onlygen90_tr, '../out/gpc_genetic_opcrit90_train.txt', sep='\t')
fwrite(onlygen90_ts, '../out/gpc_genetic_opcrit90_test.txt', sep='\t')

# Save name of features to scale
rescalevar90 <- sort(c(c_items[which(c_items != "DIPAD.AGE")], 
                  oc_items[which(oc_items != "OPCRIT.90")], 
                  n_items, prs_var, pc_vars))
fwrite(data.table("feat"=rescalevar90), 
       '../out/rescale_feature_opcrit90.txt')

# Save categorical variable indices
catvar90 <- which(names(clinical90_tr)[1:(length(names(clinical90_tr))-1)]
                  %in% sort(c(bc_items, cat_items, 
                              names(clinical90_tr)[grep("_[0-9]", 
                                                        names(clinical90_tr))]))) - 1
fwrite(data.table("idx"=catvar90), 
       '../out/categorical_feature_gpc_idx.txt')


