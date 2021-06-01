require(data.table)
require(boot)
require(diffuStats)
require(ggpubr)


auprc <- function(data, i) {
  return(metric_auc(data[i,]$label, data[i,]$pred_prob, curve = 'PRC'))
}

# Create data table df with all outcomes, ancestries, and features

df_auprc <- data.table(AUPRC = numeric(), outcome = factor(),
                       ancestry = factor(), feat = factor())
for (f in unique(df$feat)) {
  for (a in unique(df$ancestry)) {
    for (o in unique(df$outcome)) {
      subs <- df[feat == f & ancestry == a & outcome == o]
      set.seed(42)
      bootstrap_auprc <- boot(subs, auprc, strata = subs$label, R = 100)
      tmp <- as.data.table(bootstrap_auprc$t)
      names(tmp) <- 'AUPRC'
      tmp$outcome <- rep(o, nrow(tmp))
      tmp$ancestry <- rep(a, nrow(tmp))
      tmp$feat <- rep(f, nrow(tmp))
      ls <- list(df_auprc, tmp)
      df_auprc <- rbindlist(ls)
    }
  }
}


df_auprc$myfactor <- paste(df_auprc$outcome, df_auprc$ancestry, df_auprc$feat, sep = '-')
ptestmat <- pairwise.t.test(df_auprc$AUPRC, df_auprc$myfactor, p.adjust.method = 'fdr')$p.value

stats <- as.data.frame(tapply(df_auprc$AUPRC, df_auprc$myfactor, mean))
stats <- cbind(stats, as.data.frame(tapply(df_auprc$AUPRC, df_auprc$myfactor, sd)))
names(stats) <- c('auprc_mean', 'auprc_sd')

# Run ANOVAs
for (a in c("ALL", "AFR", "AMR", "EUR")) {
  for (o in c("agressive", "psych_admit", "opcrit90")) {
    print(sprintf("ANOVA: %s-%s", a, o))
    tmp <- df_auprc[ancestry == a & outcome == o]
    aov_mod <- aov(tmp$AUPRC ~ tmp$feat)
    print(summary(aov_mod))
  } }

