# Design III: imputed survival-time (T_I) density by method, SampleOE / SRS, replicate 1.
# VAL (true) = black, UNVAL (error-prone) = grey, each imputation method its own colour.
suppressMessages({library(mice); library(arrow); library(ggplot2)})
setwd("D:/phd-thesis/code_surv")
simdir <- "simulations/SampleOE/SRS"

te <- new.env(); load("data/True/0001.RData", envir = te); truth <- te$data
truth_ti <- data.frame(ID = truth$ID,
                       T_true = as.numeric(as.character(truth$T_I)),
                       T_star = as.numeric(as.character(truth$T_I_STAR)))

load_mids <- function(p){ e<-new.env(); load(p,envir=e); complete(get(ls(e)[1],e),1) }
load_list <- function(p){ e<-new.env(); load(p,envir=e); get(ls(e)[1],e)[[1]] }
load_parq <- function(p){ d<-as.data.frame(read_parquet(p)); if("imp_id"%in%names(d)) d<-d[d$imp_id==min(d$imp_id),]; d }

comp <- list(
  "MICE-PMM"=load_mids(file.path(simdir,"mice/0001.RData")),
  "MICE-RF" =load_mids(file.path(simdir,"rf/0001.RData")),
  "MIXGB"   =load_list(file.path(simdir,"mixgb/0001.RData")),
  "GAIN"    =load_parq(file.path(simdir,"gain/0001.parquet")),
  "SICG"    =load_parq(file.path(simdir,"sicg/0001.parquet")),
  "SIRD"    =load_parq(file.path(simdir,"sird/0001.parquet")))
mlev <- names(comp)

r0  <- comp[["MICE-PMM"]]$ID[comp[["MICE-PMM"]]$R == 0]
ref <- merge(data.frame(ID = r0), truth_ti, by = "ID")
getimp <- function(df){ df<-df[df$R==0,]; as.numeric(as.character(df$T_I)) }

rows <- list()
for (m in mlev) {
  rows[[length(rows)+1]] <- data.frame(panel=m, series="VAL",   value=ref$T_true)
  rows[[length(rows)+1]] <- data.frame(panel=m, series="UNVAL", value=ref$T_star)
  rows[[length(rows)+1]] <- data.frame(panel=m, series=m,       value=getimp(comp[[m]]))
}
pdat <- do.call(rbind, rows)
pdat$panel  <- factor(pdat$panel,  levels = mlev)
pdat$series <- factor(pdat$series, levels = c("VAL","UNVAL",mlev))

method_cols <- c("MICE-PMM"="#1B9E77","MICE-RF"="#D95F02","MIXGB"="#7570B3",
                 "GAIN"="#E7298A","SICG"="#1F78B4","SIRD"="#E6AB02")
cols <- c(VAL="black", UNVAL="grey60", method_cols)

p <- ggplot(pdat, aes(value, colour = series)) +
  geom_density(linewidth = 0.8, bounds = c(0, 24)) +
  facet_wrap(~ panel, ncol = 3) +
  scale_colour_manual(values = cols, breaks = c("VAL","UNVAL",mlev), name = NULL) +
  coord_cartesian(xlim = c(0, 24)) +
  labs(x = "Survival time", y = "Density",
       title = "Simulation Design III — Imputed Survival-Time Distribution by Method") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(face="bold", size=12),
        strip.background = element_rect(fill="grey92", colour="grey55", linewidth=0.4),
        strip.text = element_text(face="bold", size=9.5),
        panel.grid.minor = element_blank(),
        legend.position = "bottom")

ggsave("D:/phd-thesis/Shao_2026_thesis/sections/figures/simIII_TI_density.pdf",
       p, width = 11, height = 6, device = cairo_pdf)
cat("written figure\n")
