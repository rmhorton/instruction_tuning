class utils:

    import os, sys, pickle, regex
    import pandas as pd
    import numpy as np
    
    def parsify_text_column(text_column):
        import spacy as sp
        nlp = sp.load('en_core_web_trf')

        def parsify(processed, attributes=['pos_', 'dep_']):
            ptokens = []
            for token in processed:
                parts = set()
                for attrib in attributes:
                    val = token.__getattribute__(attrib).lower()
                    parts.add(val)
                ptokens.append('_'.join(parts))
            return ' '.join(ptokens)
     
        return [parsify(processed)
                for processed in nlp.pipe(text_column, n_process=12, batch_size=1000)]


    def get_item_pair_stats(item_pair_df):
        # item_pair_df must have columns named 'basket', and 'item'.
        import pandas as pd
        import sqlite3
    
        db = sqlite3.connect(":memory:")
        
        item_pair_df.to_sql("basket_item", db, if_exists="replace")
    
    
        ITEM_PAIR_STATS_QUERY = """with 
          bi as (
            select basket, item
              from basket_item
              group by basket, item  -- be sure we only count one of each kind of item per basket
          ),
          item_counts as (
            select item, count(*) item_count -- same as the number of baskets containing this item (see above)
              from bi
              group by item
          ),
          bi_count as (
            select bi.*, ic.item_count  -- basket, item, item_count
              from bi
                join item_counts ic on bi.item=ic.item
          ),
          ips as (
              select bi1.item item1, bi2.item item2,
                      bi1.item_count item1_count, bi2.item_count item2_count,
                      count(*) as both_count              
                  from bi_count bi1
                    join bi_count bi2  -- joining the table to itself
                      on bi1.basket = bi2.basket  -- two items in the same basket
                      and bi1.item != bi2.item    -- don't count the item being in the basket with itself
                  group by bi1.item, bi1.item_count, 
                           bi2.item, bi2.item_count
          ),
          cc as (
            SELECT item1, item2, item1_count, item2_count, both_count,
                  CAST(item1_count AS FLOAT)/(select count(distinct basket) from basket_item) as item1_prevalence, -- fraction of all baskets with item1
                  CAST(item2_count AS FLOAT)/(select count(distinct basket) from basket_item) as item2_prevalence, -- fraction of all baskets with item2
                  CAST(both_count AS FLOAT)/CAST(item1_count AS FLOAT) AS confidence  -- fraction of baskets with item1 that also have item2
              FROM ips
          )
        select *, confidence/item2_prevalence lift from cc
        """
    
        return pd.read_sql_query(ITEM_PAIR_STATS_QUERY, db)


    def get_nodes_and_edges_from_item_pair_stats(cooccurrence_pdf):
        """
        Convert a Pandas dataframe of item-pair statistics to separate dataframes for nodes and edges.
        """
        import pandas as pd
        from collections import Counter
        
        item_stats = {r['item1']:{'count':r['item1_count'], 'prevalence':r['item1_prevalence']} 
                        for idx, r in cooccurrence_pdf.iterrows()}
     
        item_stats.update({r['item2']:{'count':r['item2_count'], 'prevalence':r['item2_prevalence']} 
                        for idx, r in cooccurrence_pdf.iterrows()})
     
        nodes_df = pd.DataFrame([{'label':k,'count':v['count'], 'prevalence':v['prevalence']}  
                        for k,v in item_stats.items()])
        nodes_df['id'] = nodes_df.index
       
        edges_df = cooccurrence_pdf.copy()
        node_id = {r['label']:r['id'] for idx, r in nodes_df.iterrows()}
        edges_df['from'] = [node_id[nn] for nn in edges_df['item1']]
        edges_df['to'] = [node_id[nn] for nn in edges_df['item2']]
        
        print("Your graph will have {0} nodes and {1} edges.".format( len(nodes_df), len(edges_df) ))
     
        return nodes_df, edges_df[[ 'from', 'to', 'both_count', 'confidence', 'lift']]
    
    
    def get_vis_js_html(self, nodes_df, edges_df):
        """
        Generate HTML encoding vis_js graph from Pandas dataframes of nodes and edges.
        """
        nodes_str = nodes_df.to_json(orient='records')
        edges_str = edges_df.to_json(orient='records')
        
        max_weight = max(edges_df['weight'])
    
        html_string = ( 
        '     <style type="text/css">#mynetwork {width: 100%; height: 1000px; border: 3px}</style>\n'
        '     <button onclick=toggle_motion()>Toggle motion</button>\n'
        '     <div class="slidercontainer">\n'
        '            <label>minimum edge weight:\n'
        f'                <input type="range" min="0" max="{max_weight}" value="{max_weight/2}" step="{max_weight/100}" class="slider" id="min_edge_weight">\n'
        '                <input type="text" id="min_edge_weight_display" size="2">\n'
        '            </label>\n'
        '     </div>\n'
        '     <div id="mynetwork"></div>\n'
        f'     <script type="text/javascript">NODE_LIST={nodes_str};FULL_EDGE_LIST={edges_str};</script>\n'
        '     <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>\n'
        '     <script type="text/javascript">\n'
        '       const sign_color = {pos:"blue", neg:"red", zero:"black"}\n'
        '       const options = {physics:{maxVelocity: 1, minVelocity: 0.01}}\n'
        '       var edgeFilterSlider\n'
        '       var mynetwork\n'
        '       var motion_flag = false\n'
        '       function toggle_motion(){\n'
        '           motion_flag = !motion_flag\n'
        '           mynetwork.setOptions( { physics: motion_flag } )\n'
        '       }\n'
        '       function edgesFilter(edge){ return edge.value >= edgeFilterSlider.value }\n'
        '       function init_network(){\n'
        '           document.getElementById("min_edge_weight_display").value = 0.5\n'
        '           document.getElementById("min_edge_weight").onchange = function(){\n'
        '               document.getElementById("min_edge_weight_display").value = this.value\n'
        '           }\n'
        '           edgeFilterSlider = document.getElementById("min_edge_weight")\n'
        '           edgeFilterSlider.addEventListener("change", (e) => {edgesView.refresh()})\n'
        '           var container = document.getElementById("mynetwork")\n'
        '           var EDGE_LIST = []\n'
        '           for (var i = 0; i < FULL_EDGE_LIST.length; i++) {\n'
        '               var edge = FULL_EDGE_LIST[i]\n'
        '               edge["value"] = Math.abs(edge["weight"])\n'
        '               edge["title"] = "weight " + edge["weight"]\n'
        '               edge["sign"] = (edge["weight"] < 0) ? "neg" : "pos";\n'
        '               edge["color"] = {color: sign_color[edge["sign"]] };\n'
        '               edge["arrows"] = "to"\n'
        '               EDGE_LIST.push(edge)\n'
        '           }\n'
        '           var nodes = new vis.DataSet(NODE_LIST)\n'
        '           var edges = new vis.DataSet(EDGE_LIST)\n'
        '           var nodesView = new vis.DataView(nodes)\n'
        '           var edgesView = new vis.DataView(edges, { filter: edgesFilter })\n'
        '           var data = { nodes: nodesView, edges: edgesView }\n'
        '           mynetwork = new vis.Network(container, data, options)\n'
        '       }\n'
        '       init_network()\n'
        '     </script>\n'
    
        )
        return html_string
    
    
    def export_to_vis_js(self, nodes_df, edges_df, title, html_file_name):
        """
        Generate vis_js graph from Pandas dataframes of nodes and edges, and write to HTML file.
        """
        
        vis_js_html = self.get_vis_js_html(nodes_df, edges_df)
        page_html =  ('<!DOCTYPE html>\n'
            '<html lang="en">\n'
            '    <head>\n'
            f'       <title>{title}</title>\n'
            '    </head>\n'
            '    <body onload=init_network()>\n'
            f'{vis_js_html}'
            '\n'
            '    </body>\n'
            '</html>\n')
        
        with open(html_file_name, "wt") as html_file: 
            html_file.write(page_html)


    def make_cluster_node_title(row, text_df):
        title = f"{row['label']}\n({row['type']}, {row['count']} examples)"
        if row['type'] == 'instruction_cluster':
            cluster_id = row['label']
            examples = text_df[ text_df['instruction_B'] == cluster_id ]['instruction'].sample(6).values
            title += '\n' + '\n'.join(examples)
        if row['type'] == 'response_cluster':
            cluster_id = row['label']
            examples = text_df[ text_df['response_B'] == cluster_id ]['response'].sample(6).values
            title += '\n' + '\n'.join(examples)        
        return title


    def pivot_term_document_matrix_to_basket_item(tdm):
        import pandas as pd
        # This is just a simple pivot
        basket_item_rows = []
        for i, row in enumerate(instruction_ngram_counts_pdf.to_dict(orient="records")):
            for k, v in row.items():
                if v > 0:
                    basket_item_rows.append({'basket': i, "item": k})
        
        basket_item = pd.DataFrame(basket_item_rows)
        return basket_item


    def get_leiden_partition(edges):
        import leidenalg   # https://pypi.org/project/leidenalg/
        import igraph as ig
        
        edge_tuple_list = [(row['from'], row['to'], row['weight']) for row in edges[['from', 'to', 'weight']].to_dict(orient='records') ]
        G = ig.Graph.TupleList(edge_tuple_list)
        
        # G = ig.Graph.DictList( edges[['from', 'to', 'weight']].to_dict(orient='records'))
        
        leiden_partition = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition);
        return leiden_partition.membership


    def add_cluster_cols(df, embedding_col='embedding', prefix='cluster', letters='ABCDE', max_threshold=1):
        from scipy.cluster.hierarchy import ward, fcluster
        from scipy.spatial.distance import pdist
        import math
    
        # cluster the sentence vectors at various levels
        X = df[embedding_col].tolist()
        y = pdist(X, metric='cosine')
        z = ward(y)
    
        for i in range(len(letters)):
            letter = letters[i]
            col_name = f'{prefix}_{letter}'
            cluster_id = fcluster(z, max_threshold/2**i, criterion='distance')
            digits = 1 + math.floor(math.log10(max(cluster_id)))
            df[col_name] = [col_name + str(cid).zfill(digits) for cid in cluster_id]
    
        cluster_cols = [c for c in df.columns if c.startswith(f'{prefix}_')]
        return df.sort_values(by=cluster_cols)