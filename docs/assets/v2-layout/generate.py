from pathlib import Path
from html import escape
import re
import xml.etree.ElementTree as ET

repo = Path(__file__).resolve().parents[3]
assets = repo / 'docs/assets/v2-layout'
BLUE=('#eff6ff','#2563eb'); GREEN=('#ecfdf5','#059669'); ORANGE=('#fff7ed','#ea580c'); GRAY=('#f1f5f9','#64748b'); PURPLE=('#f5f3ff','#7c3aed')
class SVG:
    def __init__(self,w,h,title):
        self.w,self.h=w,h
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{escape(title)}">',f'<title>{escape(title)}</title>','''<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#64748b"/></marker></defs><style>text{font-family:Inter,"PingFang SC","Microsoft YaHei",Arial,sans-serif;fill:#172033} .small{font-size:13px} .mono{font-family:"SFMono-Regular",Consolas,monospace;font-size:13px}</style>''',f'<rect width="{w}" height="{h}" rx="14" fill="#fff"/>']
    def text(self,x,y,s,size=16,fill='#172033',anchor='start',weight='normal'):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" style="fill:{fill}">{escape(str(s))}</text>')
    def rect(self,x,y,w,h,fill='#fff',stroke='#cbd5e1',rx=10):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>')
    def line(self,points,label=None,dash=False,lx=None,ly=None):
        self.parts.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="#64748b" stroke-width="1.7" marker-end="url(#arrow)"'+(' stroke-dasharray="6 4"' if dash else '')+'/>')
        if label:
            x,y=(lx,ly) if lx is not None else (sum(p[0] for p in points)/len(points),sum(p[1] for p in points)/len(points)-7)
            self.text(x,y,label,13,'#475569',anchor='middle')
    def card(self,x,y,w,h,title,lines,color=BLUE):
        self.rect(x,y,w,h,color[0],color[1]);self.text(x+16,y+29,title,17,color[1],weight='600')
        self.parts.append(f'<line x1="{x}" y1="{y+44}" x2="{x+w}" y2="{y+44}" stroke="{color[1]}" opacity=".25"/>')
        for i,s in enumerate(lines):self.text(x+16,y+67+i*22,s,14)
    def title(self,s,sub=''):
        self.text(28,32,s,21,weight='600')
        if sub:self.text(28,57,sub,14,'#64748b')
    def save(self,name):
        text='\n'.join(self.parts+['</svg>']);ET.fromstring(text);(assets/name).write_text(text)

s=SVG(1080,630,'Layout 模块边界和数据流');s.title('01 · Layout 的三个职责','蓝色：逻辑元数据；绿色：HBM 几何；橙色：外存模板；灰色：模块外部')
s.rect(20,88,1040,355,'#f8fafc','#94a3b8');s.text(38,114,'layout 目录：不接收请求、plan 或 hash keys',16,weight='600')
s.card(40,145,280,153,'配置元数据',['KVCacheConfig → Spec/GroupInfo','原生存取标志 / kind / token span','持久 group 的 State / tail 规则'],BLUE)
s.card(400,145,280,153,'HBM 布局',['worker spec + descriptor + views','LayerView → Component → Segment','ptr / stride / payload / BlockAccess'],GREEN)
s.card(760,145,280,153,'外存布局',['GroupStoreLayout','offsets / sizes / store_bytes','通用模板 或 Block First bulk'],ORANGE)
s.line([(320,217),(400,217)],'策略',lx=360,ly=205);s.line([(680,217),(760,217)],'几何',lx=720,ly=205)
s.card(245,338,590,80,'UCMKVCacheLayout',['spec + group_layouts + store_layouts + 层索引'],PURPLE)
s.line([(180,298),(180,365),(245,365)]);s.line([(540,298),(540,338)]);s.line([(900,298),(900,365),(835,365)])
s.card(40,492,275,108,'Scheduler（外部）',['请求 / keys / block tables','生成 dispatch plan'],GRAY)
s.card(403,492,305,108,'TransferBuilder（外部）',['plan + Layout → [K,S] 矩阵','关联 key；累计 group 存储起点'],GRAY)
s.card(795,492,245,108,'Connector / Proxy',['提交 / wait / commit','实际字节搬运'],GRAY)
s.line([(315,548),(403,548)],'plan',lx=359,ly=536);s.line([(708,548),(795,548)],'Transfer',lx=751,ly=536);s.line([(555,418),(555,492)],'布局查询',lx=600,ly=467)
s.save('01-boundary.svg')

s=SVG(1240,1400,'Layout 对象关系类图');s.title('02 · 类图：引用关系与构建依赖','箭头：持有者 → 被引用对象；虚线：构建依赖。0..* 不代表不同 HBM allocation。')
cards=[
(445,85,'UCMKVCacheLayout',['spec','group_layouts / store_layouts','layer_id_by_name','layer_names_by_id'],PURPLE),
(30,85,'UCMKVCacheSpec',['groups','scheduler_block_size','ucm_cache_block_size / device_type'],BLUE),
(30,355,'UCMKVCacheGroupInfo',['group_id / kinds / is_persistent','token_block_size / tail_tokens','tail_blocks / is_eagle_group'],BLUE),
(445,355,'KVCacheGroupLayout',['layers / layer_views','base_ptrs / strides / payloads','partial 能力 / block_first','compile_access / segment_mask'],GREEN),
(860,355,'GroupStoreLayout',['access / ucm_block_offsets','store_bytes / merge_whole_blocks','build / resolve_matrices'],ORANGE),
(30,625,'UCMLayerSpec',['owner name / layer_index / native spec','storage_block_size / num_blocks','descriptor / attention_view_order'],GREEN),
(445,625,'LayerView',['layer_name / layer_id','components','segments（展开查询）'],GREEN),
(860,625,'BlockAccess',['layout（引用 GroupLayout）','block_byte_offsets [P,V]','segment_bytes [P,V]','resolve_ptrs → [K×P,M]'],GREEN),
(30,895,'TensorDescriptor',['layers / offset','layer_stride / block_stride','单位：bytes'],GREEN),
(445,895,'ComponentView',['shape / strides（元素）','segments / partial 能力','不分配或复制 KV storage'],GREEN),
(860,895,'BlockFirstView',['base_ptr / block_stride','block_size_bytes','本组连续 span，包含 page padding'],GREEN),
(445,1165,'MemorySegment',['base_ptr / block_stride_bytes','states_per_block / bytes_per_state','payload_bytes'],GREEN)]
for x,y,t,ls,c in cards:s.card(x,y,350,190,t,ls,c)
s.line([(445,170),(380,170)],'spec',lx=412,ly=160)
s.line([(205,275),(205,355)],'groups 1..*',lx=265,ly=323)
s.line([(620,275),(620,355)],'group_layouts 0..*',lx=709,ly=322)
s.line([(795,170),(1035,170),(1035,355)],'store_layouts 0..*',lx=1025,ly=155)
s.line([(445,466),(390,466),(390,593),(205,593),(205,625)],'layers 1..*',lx=310,ly=580)
s.line([(620,545),(620,625)],'views 1..*',lx=688,ly=594)
s.line([(1035,545),(1035,625)],'access',lx=1070,ly=594)
s.line([(205,815),(205,895)],'descriptor 0..1',lx=286,ly=863)
s.line([(620,815),(620,895)],'components 0..*',lx=704,ly=863)
s.line([(620,1085),(620,1165)],'segments 1..*',lx=698,ly=1133)
s.line([(795,430),(820,430),(820,305),(1228,305),(1228,990),(1210,990)],'block_first 0..1',lx=1100,ly=295)
s.line([(860,723),(815,723),(815,568),(742,568),(742,545)])
s.parts.append('<text x="808" y="705" font-size="13" transform="rotate(-90 808 705)" style="fill:#475569">layout 引用</text>')
s.line([(795,487),(836,487),(836,594),(915,594),(915,625)],'compile_access',True,lx=914,ly=584)
s.save('02-classes.svg')

s=SVG(1060,795,'Layout 初始化流程');s.title('03 · 构建顺序','阶段一：逻辑策略；阶段二：worker 绑定；阶段三：外存模板。运行时不重新解析 view。')
s.card(40,95,430,105,'parse_kv_cache_config',['原生 prefix_cacheable_group_ids / kind / tail','选择 UCM token 粒度 → 固定 tail_blocks'],BLUE)
s.card(590,95,430,105,'UCMKVCacheSpec',['scheduler 与 worker 共用策略入口','不含 device ptr 或物理 allocation 大小'],BLUE)
s.line([(470,147),(590,147)],'返回',lx=530,ly=136)
s.card(40,255,430,115,'register_kv_caches → aggregate',['输入：worker KVCacheConfig + kv_caches','UCMKVCacheLayout(..., use_layerwise)'],PURPLE)
s.card(590,255,430,115,'build_group_layouts',['descriptor 索引、逐层 native spec','storage_block_size / axis order / BF 候选'],GREEN)
s.line([(40,310),(20,310),(20,229),(805,229),(805,200)],'读取策略',True,lx=650,ly=222)
s.line([(470,310),(590,310)],'绑定',lx=530,ly=299)
s.card(590,425,430,125,'KVCacheGroupLayout → build_layer_view',['BNHC→BHNC；State 字节页；K/V unbind','真实 shape/stride → Component/Segment','展平寻址列；group partial 能力；BF span'],GREEN)
s.line([(805,370),(805,425)])
s.card(40,425,430,125,'GroupStoreLayout.build',['编译 BlockAccess 的固定范围','通用 payload 模板 或 padded BF bulk','生成 offsets/sizes/store_bytes'],ORANGE)
s.line([(590,485),(470,485)],'HBM几何',lx=531,ly=474)
s.card(40,615,980,105,'UCMKVCacheLayout 完成',['spec + group_layouts + store_layouts；建立 owner / 模型层号索引','外部再创建 UCMTransferBuilder(layout)，Layout 本身不遍历 plan'],PURPLE)
s.line([(255,550),(255,615)]);s.line([(805,550),(805,615)])
s.text(42,766,'原生禁用存取项与本地空 group 保留 native ID，不创建 HBM/store layout。',15,'#64748b')
s.save('03-init.svg')

s=SVG(1120,718,'不同 stride 的 BHNC view 与 MemorySegment');s.title('04 · 同为 [Q,2,4,1] view，物理顺序不同','示例每元素1字节；逻辑 token 范围 n1..n2。B 是逻辑块数，Q 可为 B×kernel rows。')
def strip(s,x,y,items,width=95,height=48,colors=None):
    for i,label in enumerate(items):
        fill=(colors[i] if colors else '#ecfdf5')
        s.rect(x+i*width,y,width-3,height,fill,'#94a3b8',3);s.text(x+i*width+(width-3)/2,y+29,label,14,anchor='middle')
s.text(35,111,'NHC：token 内所有 heads 连续',18,weight='600')
strip(s,35,135,['n0h0','n0h1','n1h0','n1h1','n2h0','n2h1','n3h0','n3h1'],125,48, ['#ecfdf5','#ecfdf5','#bfdbfe','#bfdbfe','#bfdbfe','#bfdbfe','#ecfdf5','#ecfdf5'])
s.rect(35,201,996,42,'#f8fafc','#059669',4);s.text(51,228,'完整 block：1 segment；部分 token n1..n2 也为一段连续 IO。',16)
s.text(35,293,'compact HNC：完整 block 内 heads 连续',18,weight='600')
strip(s,35,317,['h0n0','h0n1','h0n2','h0n3','h1n0','h1n1','h1n2','h1n3'],125,48,['#ecfdf5','#bfdbfe','#bfdbfe','#ecfdf5','#ecfdf5','#bfdbfe','#bfdbfe','#ecfdf5'])
s.rect(35,383,996,42,'#f8fafc','#ea580c',4);s.text(51,410,'完整 block：1 segment；n1..n2 分散在两个 head，当前 H>1 时拒绝部分 token。',16)
s.text(35,475,'separated heads：同一 block 的 heads 中间夹有别的内容',18,weight='600')
strip(s,35,501,['h0n0','h0n1','h0n2','h0n3'],110)
s.rect(480,501,125,48,'#f1f5f9','#94a3b8',3);s.text(542,530,'其他块/层',14,'#64748b',anchor='middle')
strip(s,620,501,['h1n0','h1n1','h1n2','h1n3'],110)
s.text(35,582,'segment 0：h0 的本块数据',16,'#059669');s.text(620,582,'segment 1：h1 的本块数据',16,'#059669')
s.text(35,624,'先按 row/head 拆段，再按地址排序并合并连续段；当前只支持整块。',16)
s.text(35,679,'group.supports_partial_tokens = all(component.supports_partial_tokens)，选择层不会放宽。',15,'#64748b')
s.save('04-views.svg')

s=SVG(1080,705,'BlockAccess 模板和广播寻址');s.title('05 · 固定模板 + block IDs → 物理 ptrs','例子：T=8，tail12，2层；base=[1000,2000]，stride=[32,16]，P=2，K=2。')
s.card(35,96,435,145,'初始化时编译 [P,V]',['starts=[4,0]   counts=[4,8]','byte offsets=[[16,8],[0,0]]','sizes=[[16,8],[32,16]]'],BLUE)
s.card(610,96,435,145,'运行时输入 [K×P]',['blocks=[3,4,8,9]','key0: b3,b4     key1: b8,b9','按 chunk 顺序排列'],GRAY)
s.line([(252,241),(252,289),(540,289)],'模板',lx=350,ly=279);s.line([(827,241),(827,289),(540,289)],'实际块号',lx=722,ly=279)
s.card(170,322,740,126,'广播到 [K,P,M]',['base[1,1,M] + blocks[K,P,1] × stride[1,1,M]','+ fixed_offsets[1,P,M] → ptrs[K,P,M]','可选动态 offset 再换成 bytes；最后 reshape [K×P,M]'],GREEN)
s.line([(540,289),(540,322)])
s.card(35,510,485,137,'key0：两块 × 两列',['b3: [1112,2056]   （只取后4token）','b4: [1128,2064]   （完整8token）','1000+3×32+16 = 1112'],GREEN)
s.card(560,510,485,137,'key1：两块 × 两列',['b8: [1272,2136]   （只取后4token）','b9: [1288,2144]   （完整8token）','2000+9×16 = 2144'],GREEN)
s.line([(325,448),(325,510)]);s.line([(755,448),(755,510)])
s.text(35,683,'reshape 表达块与模板的重复关系；不移动 HBM KV bytes。',15,'#64748b')
s.save('05-access.svg')

s=SVG(1120,645,'HBM 指针与外存 offset 的区别');s.title('06 · HBM 的绝对地址 ≠ 外存的相对 offset','继续使用 tail12 示例。group 内顺序：旧 block 的 L0/L1，再是新 block 的 L0/L1。')
s.card(35,98,475,190,'HBM：本次源地址（key0）',['旧 b3 L0: ptr1112，size16','旧 b3 L1: ptr2056，size8','新 b4 L0: ptr1128，size32','新 b4 L1: ptr2064，size16'],GREEN)
s.card(610,98,475,190,'外存：本组固定模板',['sizes=[[16,8],[32,16]]','cumsum=[16,24,56,72]','减当前size → offsets=[0,16,24,56]','store_bytes=72'],ORANGE)
s.line([(510,195),(610,195)],'映射',lx=560,ly=183)
s.text(35,344,'一个 key 的本组外存值：',18,weight='600')
# proportional byte lanes, 12px per byte
x=35;y=372
for label,n,c in [('旧L0 16B',16,'#bfdbfe'),('旧L1 8B',8,'#fde68a'),('新L0 32B',32,'#a7f3d0'),('新L1 16B',16,'#fed7aa')]:
    s.rect(x,y,n*12-2,65,c,'#94a3b8',3);s.text(x+n*6,y+37,label,15,anchor='middle');x+=n*12
for n in [0,16,24,56,72]:s.text(35+n*12,459,n,14,anchor='middle')
s.card(35,497,1050,99,'只选择 L1：不重新打包',['offsets=[16,56]；sizes=[8,16]；ptrs=[2056,2064]；本组 store_bytes 仍为72','下一个 group 的起点由外部 TransferBuilder 加72，而不是按本次 IO 字节数加24。'],ORANGE)
s.save('06-store.svg')

s=SVG(1120,720,'Block First bulk 与紧凑 layerwise 的存储差异');s.title('07 · Block First：bulk 与 layerwise 使用不同外存模板','例子：L0有效32B/page48B；L1有效16B/page32B；本组span80B；全局stride96B。')
s.text(35,109,'HBM 一块：',18,weight='600')
x=35
for label,n,c in [('L0 32B',32,'#bfdbfe'),('pad16',16,'#e2e8f0'),('L1 16B',16,'#a7f3d0'),('pad16',16,'#e2e8f0'),('组外16',16,'#f1f5f9')]:
    s.rect(x,134,n*10-2,62,c,'#94a3b8',3);s.text(x+n*5,169,label,15,anchor='middle');x+=n*10
s.text(35,220,'0',14);s.text(355,220,'32',14);s.text(515,220,'48',14);s.text(675,220,'64',14);s.text(835,220,'80',14);s.text(995,220,'96',14)
s.card(35,270,485,190,'bulk：use_layerwise=False',['有 BlockFirstView 且访问完整块','外存 offset0，复制本组80B','包含两处 page padding','组外16B不存；每块1段 IO'],ORANGE)
s.card(590,270,485,190,'layerwise：use_layerwise=True',['始终使用通用有效 payload 模板','L0：offset0，size32','L1：offset32，size16','一块总外存48B；不保留page间隙'],BLUE)
s.line([(275,196),(275,270)]);s.line([(750,196),(830,245),(830,270)])
s.card(35,517,1040,136,'mask 与存储格式是两个不同概念',['配置 bulk 后临时只选 L1：它仍写 bulk 模板的 offset48。','配置 layerwise 后选 L1：它写紧凑模板的 offset32。','use_layerwise 在初始化时决定格式；segment_mask 只选择本次访问列。'],PURPLE)
s.text(35,688,'State 也可用 padded bulk；NPU 不建立整组 BlockFirstView。r7 与 layerwise 后缀隔离格式。',15,'#64748b')
s.save('07-block-first.svg')

s=SVG(1080,615,'State 字节页和通用寻址');s.title('08 · State 与 Attention 共用地址公式','布局解析负责内容/步长；恢复目标 A/B、原生 copy 与等待时机由模块外部处理。')
s.card(35,99,465,126,'CUDA State',['已注册 [B,1,1,content_bytes] int8 view','states_per_block=1','保留实际 data_ptr/block stride'],GREEN)
s.card(580,99,465,126,'Ascend State',['typed components → _as_byte_page','[Q,1,1,每 row 有效字节]','dtype view + as_strided；不复制缓存'],GREEN)
s.card(210,293,660,125,'MemorySegment → BlockAccess',['ptr = base + block_id × block_stride + byte_offset','一个完整快照：t×1/T 只有完整边界可整除','整组 Block First 时允许保存已有 page padding'],ORANGE)
s.line([(267,225),(267,259),(400,259),(400,293)]);s.line([(812,225),(812,259),(680,259),(680,293)])
s.card(35,485,1010,94,'Layout 外部：生命周期',['CUDA 填 A+B；Ascend 先填边界 A 再原生 A→B；保存使用精确 boundary_state_offloads','当前 State 在 forward 前等完。State layerwise 加载/计算流水仍待实现。'],GRAY)
s.line([(540,418),(540,485)],'布局供查询',lx=614,ly=458)
s.save('08-state.svg')

# A small offline Markdown renderer for this document: images are SVG files,
# source diagrams remain editable code blocks, all assets are local.
md = (repo/'docs/connector-v2-layout-design.md').read_text()
def inline(t):
    t=escape(t)
    t=re.sub(r'`([^`]+)`',lambda m:'<code>'+m[1]+'</code>',t)
    t=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',t)
    t=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:f'<a href="{m[2]}">{m[1]}</a>',t)
    return t
lines=md.splitlines();chunks=[];toc=[];i=0
while i<len(lines):
    line=lines[i]
    if not line.strip():i+=1;continue
    if line.startswith('```'):
        lang=line[3:].strip();i+=1;code=[]
        while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
        rendered='<pre><code>'+escape('\n'.join(code))+'</code></pre>'
        chunks.append('<details><summary>查看 Mermaid 类图源码</summary>'+rendered+'</details>' if lang=='mermaid' else rendered);i+=1;continue
    m=re.match(r'^(#{1,6}) (.+)$',line)
    if m:
        level=len(m[1]);ident='section-'+str(len(toc)) if level==2 else 'heading-'+str(i)
        chunks.append(f'<h{level} id="{ident}">{inline(m[2])}</h{level}>')
        if level==2:toc.append((ident,m[2]))
        i+=1;continue
    m=re.match(r'^!\[([^\]]*)\]\(([^)]+)\)$',line)
    if m:
        chunks.append(f'<figure><button class="zoom" data-src="{m[2]}" data-alt="{escape(m[1])}">放大图</button><img loading="lazy" src="{m[2]}" alt="{escape(m[1])}"><figcaption>{escape(m[1])}</figcaption></figure>');i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            cells=[c.strip() for c in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',c) for c in cells):rows.append(cells)
            i+=1
        head='<thead><tr>'+''.join('<th>'+inline(c)+'</th>' for c in rows[0])+'</tr></thead>'
        body='<tbody>'+''.join('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in row)+'</tr>' for row in rows[1:])+'</tbody>'
        chunks.append('<div class="table-wrap"><table>'+head+body+'</table></div>');continue
    if re.match(r'^(- |\d+\. )',line):
        ordered=bool(re.match(r'^\d+\. ',line));tag='ol' if ordered else 'ul';items=[]
        while i<len(lines) and re.match(r'^(- |\d+\. )',lines[i]):
            items.append('<li>'+inline(re.sub(r'^(- |\d+\. )','',lines[i]))+'</li>');i+=1
        chunks.append('<'+tag+'>'+''.join(items)+'</'+tag+'>');continue
    paragraph=[line];i+=1
    while i<len(lines) and lines[i].strip() and not re.match(r'^(#|```|\||!\[|- |\d+\. )',lines[i]):paragraph.append(lines[i]);i+=1
    chunks.append('<p>'+inline(' '.join(paragraph))+'</p>')
css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f5f7fb;color:#172033;font:16px/1.8 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}nav{position:fixed;top:0;left:0;bottom:0;width:252px;padding:26px 18px;background:#0f172a;color:#e2e8f0;overflow:auto}nav strong{display:block;font-size:19px;margin-bottom:12px}nav a{display:block;color:#cbd5e1;font-size:13px;line-height:1.6;padding:8px 6px;text-decoration:none;border-radius:5px}nav a:hover{background:#1e293b;color:#fff}main{margin-left:252px;padding:40px 46px 80px;max-width:1500px}article{max-width:1150px;margin:auto;background:white;border:1px solid #e2e8f0;border-radius:16px;padding:36px}h1{font-size:30px;line-height:1.4;margin:0 0 22px}h2{font-size:24px;margin-top:48px;padding-top:10px;border-top:1px solid #e2e8f0;scroll-margin-top:20px}h3{font-size:19px;margin-top:28px}p,li{line-height:1.85}li{margin:7px 0}a{color:#2563eb}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.87em;background:#f1f5f9;padding:2px 5px;border-radius:4px}pre{background:#0f172a;color:#e2e8f0;padding:20px;border-radius:10px;overflow:auto;line-height:1.7}pre code{background:none;padding:0;color:inherit;font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #e2e8f0;padding:10px 12px;text-align:left;vertical-align:top}th{background:#eff6ff}tr:nth-child(even){background:#f8fafc}.table-wrap{overflow:auto;margin:20px 0}figure{position:relative;margin:25px 0;padding:15px 10px 10px;background:#fff;border:1px solid #e2e8f0;border-radius:12px}figure img{display:block;width:100%;height:auto}figcaption{font-size:13px;text-align:center;color:#64748b;margin-top:8px}.zoom{float:right;border:1px solid #cbd5e1;background:#f8fafc;color:#334155;padding:6px 10px;border-radius:6px;cursor:pointer}details{background:#f8fafc;padding:16px;border:1px solid #e2e8f0;border-radius:8px}summary{cursor:pointer;font-weight:600}dialog{width:96vw;height:95vh;border:1px solid #94a3b8;border-radius:12px;padding:16px;overflow:auto}dialog::backdrop{background:#0f172acc}.dialog-tools{position:sticky;top:0;display:flex;gap:10px;padding:6px;background:#fff;z-index:1}.dialog-tools button{padding:7px 12px;background:#f1f5f9;border:1px solid #cbd5e1;border-radius:5px;cursor:pointer}dialog img{display:block;max-width:none;width:100%;height:auto}.caption{color:#64748b;font-size:13px}footer{margin-top:32px;color:#64748b;font-size:13px}@media(max-width:900px){nav{position:relative;width:auto}nav a{display:inline-block;margin:4px}main{margin:0;padding:18px}article{padding:20px}}@media print{nav,.zoom,dialog{display:none}main{margin:0;padding:0}article{border:0;padding:0;max-width:none}h2{break-before:auto}figure,table,pre{break-inside:avoid}body{background:#fff;font-size:11pt}a{color:inherit}pre{white-space:pre-wrap;background:#f1f5f9;color:#172033}}'''
js='''const dialog=document.querySelector('dialog');const target=dialog.querySelector('img');let scale=100;document.querySelectorAll('.zoom').forEach(b=>b.addEventListener('click',()=>{target.src=b.dataset.src;target.alt=b.dataset.alt;scale=100;target.style.width=scale+'%';dialog.showModal()}));document.querySelector('#close').onclick=()=>dialog.close();document.querySelector('#plus').onclick=()=>{scale=Math.min(300,scale+25);target.style.width=scale+'%'};document.querySelector('#minus').onclick=()=>{scale=Math.max(50,scale-25);target.style.width=scale+'%'};document.querySelector('#reset').onclick=()=>{scale=100;target.style.width='100%'};'''
html='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UCM v2 Layout 详细设计</title><style>'+css+'</style></head><body><nav><strong>UCM · Layout v2</strong><div class="caption">源码设计 · 2026-10-09<br>离线图文版 · 8张SVG图</div>'+''.join(f'<a href="#{id}">{escape(t)}</a>' for id,t in toc)+'</nav><main><article>'+''.join(chunks)+'<footer>本文与 Markdown 版内容一致。无需联网，可用浏览器打印为 PDF。</footer></article></main><dialog><div class="dialog-tools"><button id="close">关闭</button><button id="minus">−</button><button id="reset">100%</button><button id="plus">＋</button></div><img alt="放大的设计图"></dialog><script>'+js+'</script></body></html>'
(repo/'docs/connector-v2-layout-design.html').write_text(html)
print('已生成8张SVG，以及完整离线HTML图文版。')
