require(data.table)
require(reshape2)
require(stringr)
require(ggpubr)

data <- data.table(f2 = numeric(), outcome = factor(), feat = factor(), ancestry = factor())
for (data_name in c('biome_agressive', 'biome_psych_admit', 'opcrit90')) {
  for (feat in c('all', 'clinical', 'genetic')) {
    for (a in c('ALL', 'AFR', 'AMR', 'EUR')) {
      if (a == 'ALL') {
        df <- fread(sprintf("../out/gridsearch_LRscores.txt",
                            data_name, feat), sep = ',', header = TRUE) } else {
        # df <- fread(sprintf("../ancestries/%s/out/out_%s/%s/gridsearch_LRscores.txt",
        #                     a, data_name, feat), sep = ',', header = TRUE)
      }
      selcol <- names(df)[grep('split[0-9]*_test_F2', names(df))]
      df <- df[mean_test_F2 == max(df$mean_test_F2), ..selcol]
      df <- melt(df, value.name = "f2")
      df$outcome <- rep(data_name, nrow(df))
      df$feat <- rep(feat, nrow(df))
      df$ancestry <- rep(a, nrow(df))
      df <- subset(df, select = -variable)
      data <- rbind(data, df) } } }

data$myfact <- paste(data$outcome, data$feat, data$ancestry, sep = '-')
# pairwise.t.test(data$f2, data$myfact, p.adjust.method = 'fdr')
ttest <- compare_means(f2 ~ myfact, data = data, p.adjust.method = 'fdr')
ttest$p.adj.signif <- ttest$p.signif
ttest$p.adj.signif[which(ttest$p.adj >= 0.05)] <- 'ns'
ttest$p.adj.signif[which(ttest$p.adj < 0.05)] <- '*'
ttest$p.adj.signif[which(ttest$p.adj < 0.01)] <- '**'
ttest$p.adj.signif[which(ttest$p.adj < 0.001)] <- '***'
ttest$p.adj.signif[which(ttest$p.adj < 0.0001)] <- '****'

for (o in c('biome_agressive', 'biome_psych_admit', 'opcrit90')) {
  for (a in c('ALL', 'AFR', 'AMR', 'EUR')) {
    data_rid <- data[outcome == o & ancestry == a]
    ttest_rid <- ttest[grep(sprintf('%s-[a-z]*-%s', o, a), ttest$group1),]
    ttest_rid <- ttest_rid[grep(sprintf('%s-[a-z]*-%s', o, a), ttest_rid$group2),]
    ttest_rid$group1 <- as.vector(sapply(ttest_rid$group1, function(x) str_split(x, '-')[[1]][2]))
    ttest_rid$group2 <- as.vector(sapply(ttest_rid$group2, function(x) str_split(x, '-')[[1]][2]))
    print(sprintf("%s-%s", o, a))
    print(ttest_rid)
    ttest_rid <- ttest_rid[ttest_rid$p.adj.signif < 0.05,]
    ttest_rid <- ttest_rid %>% mutate(y.position = seq(from = max(data_rid$f2) + 0.05,
                                                       to = 1.15, by = 0.06)[1:nrow(ttest_rid)])
    if (max(ttest_rid$y.position) >= 0.99) {
      lim <- max(ttest_rid$y.position) + 0.1 } else {
      lim <- 1
    }
    options(repr.plot.width = 1, repr.plot.height = 0.75)
    print(ggplot(data_rid, aes(x = as.factor(feat), y = f2)) +
            stat_boxplot(geom = 'errorbar', width = 0.1) +
            geom_boxplot(fill = 'white') +
            scale_y_continuous(sec.axis = dup_axis(label = NULL,
                                                   name = NULL),
                               expand = expansion(mult = c(0, 0)),
                               breaks = pretty(c(0, 1), n = 10),
                               limits = c(-0.01, lim)) +
            ylab("F2") +
            xlab("") +
            stat_summary(fun = mean, geom = "point", shape = 21, size = 2) +
            stat_pvalue_manual(ttest_rid, label = "p.adj.signif")) } }


# Run ANOVAs
for (a in c("ALL", "AFR", "AMR", "EUR")) {
  for (o in c("biome_agressive", "biome_psych_admit", "opcrit90")) {
    print(sprintf("ANOVA: %s-%s", a, o))
    tmp <- data[ancestry == a & outcome == o]
    aov_mod <- aov(tmp$f2 ~ tmp$feat)
    print(summary(aov_mod))
  } }
