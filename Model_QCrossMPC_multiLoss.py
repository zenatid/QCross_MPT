from torch.nn import LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import logging
from Codes import sign_to_bin
import numpy as np
from losses import *
from SupConLoss import *
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def _incidence_to_mask(mat: torch.Tensor) -> torch.Tensor:
    """Return boolean mask [1,*,*] for attn  (True = BLOCK)."""
    # mat: [M,N] with 1 where edge exists
    mask = ~(mat.bool()).unsqueeze(0).unsqueeze(0)  # [1,1,M,N]
    return mask

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        if N > 1:
            self.norm2 = LayerNorm(layer.size)

    def forward(self, x, x2, mask_VN, mask_CN):
        layer_outputs = []
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, x2, mask_VN)
            x2 = layer(x2, x, mask_CN)
            if idx == len(self.layers) // 2 and len(self.layers) > 1:
                x = self.norm2(x)
                x2 = self.norm2(x2)
            layer_outputs.append(torch.cat([x, x2], dim=1))
        return self.norm(x), self.norm(x2), layer_outputs

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, x2, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x2, x2, mask))
        return self.sublayer[1](x, self.feed_forward)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

#################################
@torch.no_grad()
def logical_class_from_true_err(eps_true, L):
    # version 1: cast to float
    parity = (L.float() @ eps_true.float().T) % 2        # [k,B] float 0/1
    parity = parity.long().T                              # [B,k] long
    bits   = 2 ** torch.arange(L.size(0), device=L.device, dtype=torch.long)
    return (parity * bits).sum(-1)                        # [B]  long


def info_nce(z: torch.Tensor, class_id: torch.Tensor, t: float = 0.1) -> torch.Tensor:
    """
    Supervised InfoNCE loss with self-term removed from both numerator and denominator.
    Args
    ----
    z         : [B, d] float tensor – feature vectors.
    class_id  : [B]    long / int tensor – logical-class labels.
    t         : temperature τ (default 0.1).

    Returns
    -------
    scalar loss (torch.Tensor, shape []).
    """

    # 1. Ensure unit-norm embeddings (safeguard in case caller forgot).
    z = F.normalize(z, dim=-1)                # [B, d]

    # 2. Pairwise similarities.
    sim = torch.matmul(z, z.T)                # [B, B], cosine in [-1, 1]
    sim.div_(t)                               # divide by τ

    # 3. Numerical stabilisation – subtract row-wise max before exp.
    sim = sim - sim.max(dim=1, keepdim=True).values

    # 4. Masks.
    B      = z.size(0)
    diag   = torch.eye(B, dtype=torch.bool, device=z.device)          # self-mask
    pos_m  = class_id.unsqueeze(0).eq(class_id.unsqueeze(1)) & ~diag  # positives ≠ self

    # 5. Remove self-similarity everywhere, exponentiate.
    exp_sim = torch.exp(sim).masked_fill(diag, 0.)   # [B, B]

    # 6. Numerator and denominator.
    pos_exp = torch.where(pos_m, exp_sim, torch.zeros_like(exp_sim)).sum(dim=1)  # Σ_{j∈P(i)}
    denom   = exp_sim.sum(dim=1) + 1e-9                                          # Σ_{k≠i}

    # 7. Compute per-anchor loss, skip anchors without positives.
    loss_vec = -torch.log((pos_exp + 1e-9) / denom)
    valid    = pos_m.any(dim=1)                               # at least one positive
    return loss_vec[valid].mean() if valid.any() else z.new_tensor(0.0)

#################################
#################################

#########################
####### Model  ##########
#########################

class ECC_Transformer(nn.Module):
    def __init__(self, args, dropout=0):
        super(ECC_Transformer, self).__init__()
        ####
        self.no_g = args.no_g
        code = args.code
        self.pc_matrix = code.pc_matrix
        self.logic_matrix = code.logic_matrix
        self.n_phys_qubits = code.n

        c = copy.deepcopy
        attn = MultiHeadedAttention(args.h, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_model * 4, dropout)

        # positional encodings
        self.src_embed_VN = torch.nn.Parameter(torch.empty(
            (code.n, args.d_model)))
        self.src_embed_CN = torch.nn.Parameter(torch.empty(
            (code.pc_matrix.size(0), args.d_model)))
        self.src_embed_LP = torch.nn.Parameter(torch.empty(
            (code.logic_matrix.size(0), args.d_model)))

        # ---------------------------------------------------------------------
        # inside ECC_Transformer.__init__(...)
        #     n      = number of variable-nodes  (physical qubits)
        #     m      = number of check-nodes     (syndrome bits)
        #     k      = number of logical qubits  (= code.logic_matrix.size(0))
        # ---------------------------------------------------------------------
        # 1. construct the incidence matrix that already contains the
        #    extra logical-parity rows
        M_checks = code.pc_matrix.bool()  # shape [m , n]   (H)
        M_logic = code.logic_matrix.bool()  # shape [k , n]   (L)
        M_stack = torch.cat([M_checks, M_logic], dim=0)  # [m+k , n]

        m_k = M_stack.size(0)  # total CN+LP tokens
        device = M_stack.device

        # 2. build masks  ------------------------------------------------------
        if args.no_mask:  # <-- command-line flag to disable masking
            self.src_mask_VN = None  # Let attention see everybody
            self.src_mask_CN = None
        else:
            # -- convert incidence to boolean mask ---------------------------------
            # VN query  ->  CN+LP key
            #   mask_VN[q,k]  == True  ⟹  *forbid* attention VN_q -> CNLP_k
            self.src_mask_VN = (~M_stack.T).unsqueeze(0).unsqueeze(0)
            # [1, 1, n   , m+k]

            # CN+LP query  ->  VN key
            self.src_mask_CN = (~M_stack).unsqueeze(0).unsqueeze(0)
            # [1, 1, m+k , n]

            # move to same device / dtype as rest of model
            self.src_mask_VN = self.src_mask_VN.to(device)
            self.src_mask_CN = self.src_mask_CN.to(device)

        print('mask VN ', None if self.src_mask_VN is None
        else self.src_mask_VN.shape,
              'mask CN ', None if self.src_mask_CN is None
              else self.src_mask_CN.shape)
        # ---------------------------------------------------------------------

        k = code.logic_matrix.size(0)          # # logical qubits (2 for toric code)
        m = code.pc_matrix.size(0)             # # check operators
        self.lp_head = nn.Sequential(          # <- NEW head
            nn.Linear(m, 4*m),
            nn.GELU(),
            nn.Linear(4*m, k)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # pools over the 'node' axis

        self.N_size = args.N_dec

        self.decoder = Encoder(EncoderLayer(
            args.d_model, c(attn), c(ff), dropout), args.N_dec)

        self.oned_final_embed = torch.nn.Sequential(
            *[nn.Linear(args.d_model, 1)])
        #self.out_fc = nn.Linear(code.n + code.pc_matrix.size(0), code.n)
        self.out_fc = nn.Linear(args.d_model, code.n)

        ## InfoNCE loss ##
        #nn.Linear(args.d_model, args.d_model, bias=False)
        self.contrastive_proj = clones(nn.Linear(args.d_model, args.d_model, bias=False), args.N_dec)
        self.ber_proj = clones(torch.nn.Sequential(*[nn.Linear(args.d_model, 1)]), args.N_dec)
        self.oned_final_proj = clones(nn.Linear(code.n + code.pc_matrix.size(0), code.n), args.N_dec)
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.07)))
        ##################

        #
        N_in = 5
        non_lin_fun = torch.nn.GELU
        layers = [torch.nn.Linear(code.pc_matrix.size(0), N_in*code.n), non_lin_fun()]
        for _ in range(1):
            layers += [torch.nn.Linear(N_in*code.n, N_in*code.n),non_lin_fun()]
        layers += [torch.nn.Linear(N_in*code.n, code.n)]
        #
        self.syn_to_noise = torch.nn.Sequential(*layers)
        #

        #self.get_mask(code)
        if args.no_mask > 0:
            self.src_mask = None
        logging.info(f'Mask:\n {self.src_mask_VN}')
        ###
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.magnitude_pred = []

    def _build_masks(self, code, force_dense=False):
        if force_dense:
            self.register_buffer('src_mask_VN', None)
            self.register_buffer('src_mask_CN', None)
            return
        # VN↔CN incidence
        M_vn_cn = code.pc_matrix.clone()                          # [m,n]
        # VN↔LP incidence  (logic matrix acts on physical qubits)
        M_vn_lp = code.logic_matrix.clone()                       # [k,n]

        # Full set of "check-side tokens" = [CN • LP]     (size m+k)
        M_stack = torch.cat([M_vn_cn, M_vn_lp], dim=0)            # [(m+k), n]

        # Masks for the two attention calls in the *bi-partite* layer
        self.register_buffer('src_mask_VN', _incidence_to_mask(M_stack.t())) # VN queries
        self.register_buffer('src_mask_CN', _incidence_to_mask(M_stack))     # CN+LP query

    @torch.no_grad()
    def _logical_parity(self, synd: torch.Tensor) -> torch.Tensor:
        """
        Predict   \hat b_L  from syndrome.  Returns hard 0/1 bits  [B,k].
        """
        logits = self.lp_head(synd)            # [B,k]
        return (logits.sigmoid() > 0.5).float()                                  # [B,k]

    def forward(self, magnitude, syndrome):
        """magnitude: dummy placeholder (ignored); syndrome: [B,m]."""
        # (0) Predict qubit reliabilities (syn_to_noise)
        magnitude = self.syn_to_noise(syndrome) # [B,n] logits
        if self.no_g:
            magnitude = magnitude*0+1
        self.magnitude_pred.append(magnitude)

        self.syndrome = syndrome
        if self.no_g:  # ablation flag
            magnitude = magnitude * 0 + 1.

        # (1) Build three token streams -------------------------------------------------
        VN = self.src_embed_VN.unsqueeze(0) * magnitude.unsqueeze(-1)  # [B,n,d]
        CN = self.src_embed_CN.unsqueeze(0) * syndrome.unsqueeze(-1)  # [B,m,d]

        b_L = self._logical_parity(syndrome)  # [B,k]
        LP = self.src_embed_LP.unsqueeze(0) * b_L.unsqueeze(-1)  # [B,k,d]

        # Stack CN and LP so that part-2 of bi-attn gets (m+k) tokens
        CNLP = torch.cat([CN, LP], dim=1)  # [B,m+k,d]

        # (2) Cross-message passing  VN ↔ CNLP
        emb_VN, emb_CNLP, inter = self.decoder(VN, CNLP, self.src_mask_VN, self.src_mask_CN)
        # 1. after you have the node-embeddings:
        #    emb_VN : [B, n, d]
        #    emb_CNLP : [B, m + k, d]
        #    --> concatenate on node axis
        nodes = torch.cat([emb_VN, emb_CNLP], dim=1)  # [B, n + m + k, d]

        # 2. transpose so that AdaptiveAvgPool1d pools over nodes
        nodes = nodes.transpose(1, 2)  # [B, d, n + m + k]
        pooled = self.pool(nodes).squeeze(-1)  # [B, d]

        # 3. final projection
        out = self.out_fc(pooled)  # [B, n]

        return out, inter, b_L  # <- return LP parity for auxiliary losses

    def loss(self, z_pred, z2, emb_layers, b_L):
        #dealing with DP activations
        indices = [xx.device.index for xx in self.magnitude_pred]
        if len(list(set(indices))) > 1:
            device_zero = self.magnitude_pred[np.where(np.array(indices)==0)[0][0]]
            self.magnitude_pred = torch.cat([self.magnitude_pred[np.where(np.array(indices)==ii)[0][0]].to(device_zero) for ii in range(len(self.magnitude_pred))],0)
        else:
            self.magnitude_pred = self.magnitude_pred[0]
        ####
        class_id = logical_class_from_true_err(z2, self.logic_matrix).to(z2.device)
        ####
        loss_ber_reg = 0.0
        loss_ssl = 0.0
        l = 1.0
        for emb, contrastive_head, ber_head, oned_head in zip(emb_layers,self.contrastive_proj, self.ber_proj, self.oned_final_proj):
            # z_inter = oned_head(ber_head(emb).squeeze(-1))
            # loss_ber_reg += F.binary_cross_entropy_with_logits(-z_inter, 1 - z2)
            # emb_proj = F.normalize(contrastive_head(emb).mean(dim=1),dim=-1)
            # loss_ssl += info_nce(emb_proj, class_id)
            l += 1.0
        loss1 = F.binary_cross_entropy_with_logits(z_pred, 1-z2)
        loss2 = F.binary_cross_entropy_with_logits(self.magnitude_pred, 1-z2)

        #lp_logits = ( self.logic_matrix.float() @ z_pred.T ).T         # [B,k] float
        loss_lp = F.binary_cross_entropy_with_logits(
            self.lp_head(self.syndrome),  # [B,k] logits
            (self.logic_matrix.float() @ z2.T).T)

        ###
        self.magnitude_pred = []
        return loss1,loss2, loss_ssl, loss_ber_reg, loss_lp


    def get_mask(self, code, no_mask=False):
        if no_mask:
            self.src_mask = None
            return

        def build_mask_VN(code):
            mask = torch.zeros(code.pc_matrix.size(0), code.n)
            for ii in range(code.pc_matrix.size(0)):
                idx = torch.where(code.pc_matrix[ii] > 0)[0]
                for jj in idx:
                    mask[ii, jj] += 1
            mask = mask.transpose(0, 1)
            np.savetxt('mask.txt', ~ (mask > 0), fmt='%d', delimiter=',')
            src_mask = ~ (mask > 0).unsqueeze(0).unsqueeze(0)
            return src_mask

        def build_mask_CN(code):
            mask = torch.zeros(code.pc_matrix.size(0), code.n)
            for ii in range(code.pc_matrix.size(0)):
                idx = torch.where(code.pc_matrix[ii] > 0)[0]
                for jj in idx:
                    mask[ii, jj] += 1

           # np.savetxt('mask.txt', ~ (mask > 0), fmt='%d', delimiter=',')
            src_mask = ~ (mask > 0).unsqueeze(0).unsqueeze(0)
            return src_mask

        src_mask_VN = build_mask_VN(code)
        src_mask_CN = build_mask_VN(code).transpose(-1,-2)#build_mask_CN(code)
        self.register_buffer('src_mask_VN', src_mask_VN)
        self.register_buffer('src_mask_CN', src_mask_CN)


############################################################
############################################################

if __name__ == '__main__':
    pass
