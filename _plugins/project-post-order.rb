# frozen_string_literal: true

# Keep publication dates intact while presenting project series in reading order.
module ProjectPostOrder
  def project_posts_in_order(posts)
    posts = Array(posts)
    return posts unless posts.all? { |post| post['categories']&.first == '2.PROJECT' }

    posts.sort_by do |post|
      category = post['categories'][1].to_s
      category_number = category.match(/\A2-(\d+)\./)
      number = post['title'].to_s.match(/\A(\d+)\.\s/)
      [category_number ? category_number[1].to_i : Float::INFINITY,
       category, number ? number[1].to_i : Float::INFINITY,
       post['date'].to_s, post['url'].to_s]
    end
  end
end

Liquid::Template.register_filter(ProjectPostOrder)
