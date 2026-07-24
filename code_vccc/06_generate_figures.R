suppressMessages(lapply(c("ggplot2","dplyr"), require, character.only=TRUE))
ne <- new.env(); load("./simulations/results/results_COMBINED.RData", envir=ne)
OUT <- "./simulations/results"; dir.create(OUT, recursive=TRUE, showWarnings=FALSE)

method_map  <- c(CC="CC", mice="MICE-PMM", rf="MICE-RF", mixgb="MIXGB", tabcsdi="TabCSDI", gain="GAIN", raking="RAKING", sicg="SICG", sird="SIRD")
mlev <- unname(method_map); face_vec <- ifelse(mlev %in% c("SICG","SIRD"),"bold","plain")
design_disp <- c(SRS="SRS",RS="RS",WRS="WRS",ODS_TAIL="ODS-Tail",NEYMAN_ODS="Neyman-ODS",SFS="SFS",NEYMAN_INF="Neyman-IF")
dlev <- c("SRS","RS","WRS","ODS-Tail","Neyman-ODS","SFS","Neyman-IF")
design_cols <- c("SRS"="#999999","RS"="#A6CEE3","WRS"="#1F78B4","ODS-Tail"="#B2DF8A","Neyman-ODS"="#33A02C","SFS"="#FDBF6F","Neyman-IF"="#FF7F00")
model_map <- c(LM="Gaussian", LOGISTIC="Binomial", COX="Cox")
cov_disp <- c(CD4_COUNT_BSL_sqrt_v="Baseline CD4", VL_COUNT_BSL_LOG_v="Baseline VL", AGE_AT_MED_START_v="Age at ART",
              YEAR_OF_ENROLLMENT="Enrolment yr", SEX="Sex")
phase_disp <- c("Baseline CD4"="Phase-Two Covariates","Baseline VL"="Phase-Two Covariates","Age at ART"="Phase-Two Covariates","Enrolment yr"="Phase-One Covariates","Sex"="Phase-One Covariates")
model_covs <- list(
  Gaussian = c("CD4_COUNT_BSL_sqrt_v","VL_COUNT_BSL_LOG_v","AGE_AT_MED_START_v","SEX"),
  Binomial = c("CD4_COUNT_BSL_sqrt_v","AGE_AT_MED_START_v","SEX","YEAR_OF_ENROLLMENT"),
  Cox      = c("CD4_COUNT_BSL_sqrt_v","AGE_AT_MED_START_v","SEX","YEAR_OF_ENROLLMENT"))
LAD <- c(5,10,15,20,30,40,50,75,100,150,200,300,500,1000,2000,5000); nicec <- function(x){ v<-LAD[which(LAD>=x)[1]]; if(is.na(v)) tail(LAD,1) else v }

d <- ne$combined_long
d$Coeff <- suppressWarnings(as.numeric(d$Coeff))
d$Method <- as.character(d$Method); d$Design <- toupper(as.character(d$Design)); d$Covariate <- as.character(d$Covariate); d$Model <- as.character(d$Model)
truth <- d %>% filter(Method=="VAL") %>% group_by(Model, Covariate) %>% summarise(tv=mean(Coeff, na.rm=TRUE), .groups="drop")
L <- d %>% filter(Method %in% names(method_map), Design %in% names(design_disp)) %>%
  left_join(truth, by=c("Model","Covariate")) %>%
  transmute(model=unname(model_map[Model]), cov=Covariate,
            method=factor(unname(method_map[Method]), levels=mlev),
            design=factor(unname(design_disp[Design]), levels=dlev),
            b=(Coeff-tv)/tv*100) %>% filter(is.finite(b))

bstat <- function(x){ x<-x[is.finite(x)]; if(length(x)<2) rep(NA_real_,5) else boxplot.stats(x)$stats }
save_model <- function(mo, base){
  covset <- model_covs[[mo]]
  dd <- L %>% filter(model==mo, cov %in% covset) %>% mutate(cov=factor(unname(cov_disp[cov]), levels=unname(cov_disp[covset])))
  gs <- dd %>% group_by(cov,method,design) %>% summarise(s=list(bstat(b)), .groups="drop") %>%
    mutate(ymin=sapply(s,`[`,1),lower=sapply(s,`[`,2),mid=sapply(s,`[`,3),upper=sapply(s,`[`,4),ymax=sapply(s,`[`,5)) %>% select(-s) %>% filter(!is.na(mid))
  cl <- gs %>% group_by(cov) %>% summarise(clip=nicec(quantile(abs(c(lower,upper)),0.90,na.rm=TRUE)), .groups="drop")
  gd <- gs %>% left_join(cl, by="cov") %>% mutate(across(c(ymin,lower,mid,upper,ymax), ~pmax(-clip,pmin(clip,.))),
                                                 phase=factor(unname(phase_disp[as.character(cov)]), levels=c("Phase-Two Covariates","Phase-One Covariates")))
  p <- ggplot(gd, aes(method, colour=design)) +
    geom_hline(yintercept=0, linetype="dashed", colour="grey25", linewidth=0.4) +
    geom_boxplot(aes(ymin=ymin,lower=lower,middle=mid,upper=upper,ymax=ymax), stat="identity", fill=NA, linewidth=0.34, width=0.62, position=position_dodge2(padding=0.12)) +
    geom_blank(aes(y=clip)) + geom_blank(aes(y=-clip)) +
    ggh4x::facet_nested_wrap(vars(phase, cov), ncol=2, scales="free_y") +
    scale_colour_manual(values=design_cols, name="Sampling Design") +
    labs(x="Method", y="Percentage Bias (%)", title=paste0("VCCC — ", mo)) +
    theme_minimal(base_size=11) +
    theme(plot.title=element_text(face="bold", size=12),
          strip.background=element_rect(fill="grey92",colour="grey55",linewidth=0.4), strip.text=element_text(face="bold",size=9.5),
          panel.grid.minor=element_blank(), panel.grid.major.x=element_blank(),
          axis.text.x=element_text(angle=30,hjust=1,face=face_vec,size=8), legend.position="bottom") +
    guides(colour=guide_legend(nrow=1))
  ggsave(file.path(OUT,paste0(base,".pdf")), p, width=9.6, height=8); ggsave(file.path(OUT,paste0(base,".png")), p, width=9.6, height=8, dpi=200)
}
save_model("Gaussian", "vccc_gaussian_pctbias")
save_model("Binomial", "vccc_binomial_pctbias")
save_model("Cox",      "vccc_cox_pctbias")
cat("done ->", OUT, "\n")
